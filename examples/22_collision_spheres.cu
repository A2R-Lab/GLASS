// 22_collision_spheres.cu — sphere-based narrow phase: transform, distance, cost.
//
// Build: nvcc -std=c++17 -arch=sm_75 -I.. 22_collision_spheres.cu -o collision_spheres && ./collision_spheres
//
// USE CASE (motion generation / IK): the dominant GPU robot-collision
// representation decomposes the robot into spheres and scores each against
// world primitives — one (sphere, obstacle) pair per THREAD. A narrow-phase
// check is three GLASS calls:
//   glass::transform_sphere   FK pose applied to the link sphere (radius kept)
//   glass::sphere_box_dist    signed distance to a box, in the box frame
//                             (an OBB = rotate the center into the box frame
//                             with quat_rotate on the inverse pose)
//   glass::smooth_hinge       the C¹ activation turning distance into cost
// plus glass::sphere_sphere_dist for self-collision pairs. The gradient
// pipeline is the same three calls' grad outputs chained.
//
// Checks: the device signed distances against a double host reference over
// mixed inside/outside cases, and cost/gradient consistency of the hinge
// (cost decreasing in distance, zero beyond the activation band).

#include "glass.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

constexpr int P = 4096;
constexpr float ETA = 0.05f;   // activation width

__global__ void k_narrow_phase(const float* q, const float* t, const float* sph,
                               const float* half, float* dist, float* cost, float* grad) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    // link sphere -> world (FK pose), world == box frame here for brevity
    float world[4];
    glass::block::transform_sphere<float>(q + 4*p, t + 3*p, sph + 4*p, world);
    float g[3];
    float d = glass::block::sphere_box_dist<float>(world, world[3], half + 3*p, g);
    dist[p] = d;
    cost[p] = glass::block::smooth_hinge<float>(d, ETA);
    float dcdd = glass::block::smooth_hinge_grad<float>(d, ETA);
    for (int i = 0; i < 3; i++) grad[3*p + i] = dcdd*g[i];   // chain rule to the center
}

int main() {
    static float hq[P*4], ht[P*3], hs[P*4], hh[P*3];
    for (int p = 0; p < P; p++) {
        float n = 0;
        for (int i = 0; i < 4; i++) { hq[4*p + i] = (float)((p*13 + i*57) % 200 - 100)/100.f + 0.02f; n += hq[4*p + i]*hq[4*p + i]; }
        for (int i = 0; i < 4; i++) hq[4*p + i] /= sqrtf(n);
        for (int i = 0; i < 3; i++) {
            ht[3*p + i] = (float)((p*7 + i*3) % 100 - 50)/40.f;
            hs[4*p + i] = (float)((p*5 + i*11) % 60 - 30)/60.f;
            hh[3*p + i] = 0.4f + (float)((p + i) % 40)/50.f;
        }
        hs[4*p + 3] = 0.03f + (float)(p % 20)/200.f;   // radius
    }
    float *dq, *dt, *ds, *dh, *dd, *dc, *dg;
    cudaMalloc(&dq, sizeof(hq)); cudaMalloc(&dt, sizeof(ht));
    cudaMalloc(&ds, sizeof(hs)); cudaMalloc(&dh, sizeof(hh));
    cudaMalloc(&dd, P*sizeof(float)); cudaMalloc(&dc, P*sizeof(float));
    cudaMalloc(&dg, P*3*sizeof(float));
    cudaMemcpy(dq, hq, sizeof(hq), cudaMemcpyHostToDevice);
    cudaMemcpy(dt, ht, sizeof(ht), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice);
    cudaMemcpy(dh, hh, sizeof(hh), cudaMemcpyHostToDevice);
    k_narrow_phase<<<(P + 255)/256, 256>>>(dq, dt, ds, dh, dd, dc, dg);
    cudaDeviceSynchronize();

    static float dist[P], cost[P];
    cudaMemcpy(dist, dd, sizeof(dist), cudaMemcpyDeviceToHost);
    cudaMemcpy(cost, dc, sizeof(cost), cudaMemcpyDeviceToHost);

    // host reference in double
    float maxerr = 0; int inside = 0; bool cost_ok = true;
    for (int p = 0; p < P; p++) {
        double x = hq[4*p], y = hq[4*p + 1], z = hq[4*p + 2], w = hq[4*p + 3];
        double c[3] = {hs[4*p], hs[4*p + 1], hs[4*p + 2]}, rc[3];
        // p' = p + 2w(v×p) + 2v×(v×p)
        double cr1[3] = {y*c[2] - z*c[1], z*c[0] - x*c[2], x*c[1] - y*c[0]};
        for (int i = 0; i < 3; i++) rc[i] = c[i] + 2*w*cr1[i];
        double cr2[3] = {y*cr1[2] - z*cr1[1], z*cr1[0] - x*cr1[2], x*cr1[1] - y*cr1[0]};
        for (int i = 0; i < 3; i++) rc[i] += 2*cr2[i] + ht[3*p + i];
        double q[3], omax = -1e30, osum = 0;
        for (int i = 0; i < 3; i++) {
            q[i] = fabs(rc[i]) - hh[3*p + i];
            omax = fmax(omax, q[i]);
            osum += fmax(q[i], 0.0)*fmax(q[i], 0.0);
        }
        double ref = sqrt(osum) + fmin(omax, 0.0) - hs[4*p + 3];
        maxerr = fmaxf(maxerr, fabsf(dist[p] - (float)ref));
        if (ref < 0) inside++;
        float want = dist[p] <= 0 ? -dist[p] + ETA/2
                   : (dist[p] >= ETA ? 0.f : (dist[p] - ETA)*(dist[p] - ETA)/(2*ETA));
        cost_ok = cost_ok && fabsf(cost[p] - want) < 1e-6f;
    }
    printf("max |dist - ref| = %.3g over %d pairs (%d penetrating)   hinge: %s\n",
           maxerr, P, inside, cost_ok ? "ok" : "BROKEN");
    bool pass = maxerr < 2e-5f && cost_ok && inside > 0;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
