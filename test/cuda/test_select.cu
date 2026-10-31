// test_select.cu — selected-operand regression for the thread tier.
//
// A lane-parallel forward-kinematics suffix (the HJCD / GRiD pattern): each lane
// rebuilds one joint transform into a THREAD-LOCAL array through an inlined
// switch writer, then multiplies a running 4x4 product by a pointer SELECTED at
// runtime between that local array and shared memory. The product is
// glass::thread::gemm<T,4,4,4>. With `__restrict__` on the read operand of the
// compile-time gemm path, nvcc (11.4 and 13.2, sm_72 / sm_87 / sm_120) dropped
// the writer's stores to the local array once the writer was inlined, so the
// output was nondeterministic between two repeats of one launch and often NaN.
// GLASS declares no `__restrict__` on its own interfaces since 2026-10-31; this
// driver keeps that fixed (docs/agent_debugging_guide.md §1i).
//
// The writer is the GRiD-generated joint_transform_scalar for a seven-joint
// general-axis fixture (origin xyz .02 .01 .08, rpy .1 0 .03; axes y, -z, x
// prismatic, y, -z, x, y), kept in its generated shape because the fault
// depends on the optimizer seeing the stores.
//
// Usage: ./test_select <dtype f32|f64> <in.bin>
//   in.bin: float32, 2*32*N anchor values (per warp: N world transforms then
//           N local transforms, column-major 4x4 each) followed by 2*N joint
//           coordinates (per warp, one per lane).
//   Output: 2*64*16 values — two repeats of one 64-thread launch, each lane's
//           16-value result, printed from the host. Lanes >= N keep 789.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "../../glass.cuh"

constexpr int N = 7;

namespace model {
constexpr bool runtime_transform = false;
template<class T> __device__ __forceinline__ bool joint_transform_scalar(
    int joint_id, T coordinate, T *out) {
  if (runtime_transform) return false;
  switch(joint_id) {
  case 0: {
    const T theta=coordinate;
    const T s=sin(theta), c=cos(theta);
    (void)s; (void)c;
    out[0]=static_cast<T>(-0.0029945532692457677*s + 0.99955003374898754*c);
    out[1]=static_cast<T>(0.09978849497861382*s + 0.02999550020249566*c);
    out[2]=static_cast<T>(-0.99500416527802571*s);
    out[3]=static_cast<T>(0);
    out[4]=static_cast<T>(-0.029845647641081045);
    out[5]=static_cast<T>(0.99455644698403378);
    out[6]=static_cast<T>(0.099833416646828155);
    out[7]=static_cast<T>(0);
    out[8]=static_cast<T>(0.99955003374898754*s + 0.0029945532692457677*c);
    out[9]=static_cast<T>(0.02999550020249566*s - 0.09978849497861382*c);
    out[10]=static_cast<T>(0.99500416527802571*c);
    out[11]=static_cast<T>(0);
    out[12]=static_cast<T>(0.02);
    out[13]=static_cast<T>(0.01);
    out[14]=static_cast<T>(0.080000000000000002);
    out[15]=static_cast<T>(1.0);
    break;
  }
  case 1: {
    const T theta=coordinate;
    const T s=sin(theta), c=cos(theta);
    (void)s; (void)c;
    out[0]=static_cast<T>(0.029845647641081045*s + 0.99955003374898754*c);
    out[1]=static_cast<T>(-0.99455644698403378*s + 0.02999550020249566*c);
    out[2]=static_cast<T>(-0.099833416646828155*s);
    out[3]=static_cast<T>(0);
    out[4]=static_cast<T>(0.99955003374898754*s - 0.029845647641081045*c);
    out[5]=static_cast<T>(0.02999550020249566*s + 0.99455644698403378*c);
    out[6]=static_cast<T>(0.099833416646828155*c);
    out[7]=static_cast<T>(0);
    out[8]=static_cast<T>(0.0029945532692457677);
    out[9]=static_cast<T>(-0.09978849497861382);
    out[10]=static_cast<T>(0.99500416527802571);
    out[11]=static_cast<T>(0);
    out[12]=static_cast<T>(0.02);
    out[13]=static_cast<T>(0.01);
    out[14]=static_cast<T>(0.080000000000000002);
    out[15]=static_cast<T>(1.0);
    break;
  }
  case 2: {
    const T theta=coordinate;
    const T s=sin(theta), c=cos(theta);
    (void)s; (void)c;
    out[0]=static_cast<T>(0.99955003374898754);
    out[1]=static_cast<T>(0.02999550020249566);
    out[2]=static_cast<T>(0);
    out[3]=static_cast<T>(0);
    out[4]=static_cast<T>(-0.029845647641081045);
    out[5]=static_cast<T>(0.99455644698403378);
    out[6]=static_cast<T>(0.099833416646828155);
    out[7]=static_cast<T>(0);
    out[8]=static_cast<T>(0.0029945532692457677);
    out[9]=static_cast<T>(-0.09978849497861382);
    out[10]=static_cast<T>(0.99500416527802571);
    out[11]=static_cast<T>(0);
    out[12]=static_cast<T>(0.99955003374898754*theta + 0.02);
    out[13]=static_cast<T>(0.02999550020249566*theta + 0.01);
    out[14]=static_cast<T>(0.080000000000000002);
    out[15]=static_cast<T>(1.0);
    break;
  }
  case 3: {
    const T theta=coordinate;
    const T s=sin(theta), c=cos(theta);
    (void)s; (void)c;
    out[0]=static_cast<T>(-0.0029945532692457677*s + 0.99955003374898754*c);
    out[1]=static_cast<T>(0.09978849497861382*s + 0.02999550020249566*c);
    out[2]=static_cast<T>(-0.99500416527802571*s);
    out[3]=static_cast<T>(0);
    out[4]=static_cast<T>(-0.029845647641081045);
    out[5]=static_cast<T>(0.99455644698403378);
    out[6]=static_cast<T>(0.099833416646828155);
    out[7]=static_cast<T>(0);
    out[8]=static_cast<T>(0.99955003374898754*s + 0.0029945532692457677*c);
    out[9]=static_cast<T>(0.02999550020249566*s - 0.09978849497861382*c);
    out[10]=static_cast<T>(0.99500416527802571*c);
    out[11]=static_cast<T>(0);
    out[12]=static_cast<T>(0.02);
    out[13]=static_cast<T>(0.01);
    out[14]=static_cast<T>(0.080000000000000002);
    out[15]=static_cast<T>(1.0);
    break;
  }
  case 4: {
    const T theta=coordinate;
    const T s=sin(theta), c=cos(theta);
    (void)s; (void)c;
    out[0]=static_cast<T>(0.029845647641081045*s + 0.99955003374898754*c);
    out[1]=static_cast<T>(-0.99455644698403378*s + 0.02999550020249566*c);
    out[2]=static_cast<T>(-0.099833416646828155*s);
    out[3]=static_cast<T>(0);
    out[4]=static_cast<T>(0.99955003374898754*s - 0.029845647641081045*c);
    out[5]=static_cast<T>(0.02999550020249566*s + 0.99455644698403378*c);
    out[6]=static_cast<T>(0.099833416646828155*c);
    out[7]=static_cast<T>(0);
    out[8]=static_cast<T>(0.0029945532692457677);
    out[9]=static_cast<T>(-0.09978849497861382);
    out[10]=static_cast<T>(0.99500416527802571);
    out[11]=static_cast<T>(0);
    out[12]=static_cast<T>(0.02);
    out[13]=static_cast<T>(0.01);
    out[14]=static_cast<T>(0.080000000000000002);
    out[15]=static_cast<T>(1.0);
    break;
  }
  case 5: {
    const T theta=coordinate;
    const T s=sin(theta), c=cos(theta);
    (void)s; (void)c;
    out[0]=static_cast<T>(0.99955003374898754);
    out[1]=static_cast<T>(0.02999550020249566);
    out[2]=static_cast<T>(0);
    out[3]=static_cast<T>(0);
    out[4]=static_cast<T>(0.0029945532692457677*s - 0.029845647641081045*c);
    out[5]=static_cast<T>(-0.09978849497861382*s + 0.99455644698403378*c);
    out[6]=static_cast<T>(0.99500416527802571*s + 0.099833416646828155*c);
    out[7]=static_cast<T>(0);
    out[8]=static_cast<T>(0.029845647641081045*s + 0.0029945532692457677*c);
    out[9]=static_cast<T>(-0.99455644698403378*s - 0.09978849497861382*c);
    out[10]=static_cast<T>(-0.099833416646828155*s + 0.99500416527802571*c);
    out[11]=static_cast<T>(0);
    out[12]=static_cast<T>(0.02);
    out[13]=static_cast<T>(0.01);
    out[14]=static_cast<T>(0.080000000000000002);
    out[15]=static_cast<T>(1.0);
    break;
  }
  case 6: {
    const T theta=coordinate;
    const T s=sin(theta), c=cos(theta);
    (void)s; (void)c;
    out[0]=static_cast<T>(-0.0029945532692457677*s + 0.99955003374898754*c);
    out[1]=static_cast<T>(0.09978849497861382*s + 0.02999550020249566*c);
    out[2]=static_cast<T>(-0.99500416527802571*s);
    out[3]=static_cast<T>(0);
    out[4]=static_cast<T>(-0.029845647641081045);
    out[5]=static_cast<T>(0.99455644698403378);
    out[6]=static_cast<T>(0.099833416646828155);
    out[7]=static_cast<T>(0);
    out[8]=static_cast<T>(0.99955003374898754*s + 0.0029945532692457677*c);
    out[9]=static_cast<T>(0.02999550020249566*s - 0.09978849497861382*c);
    out[10]=static_cast<T>(0.99500416527802571*c);
    out[11]=static_cast<T>(0);
    out[12]=static_cast<T>(0.02);
    out[13]=static_cast<T>(0.01);
    out[14]=static_cast<T>(0.080000000000000002);
    out[15]=static_cast<T>(1.0);
    break;
  }
  default: return false;
  }
  return true;
}
} // namespace model

// The suffix: W = world[changed-1] (identity for the first joint), then
// W := W * T_j for j = changed..N-1, where T_changed is rebuilt from the lane's
// coordinate into `joint` (thread-local) and every later T_j is read from shared
// memory. `transform` is therefore a pointer SELECTED between the two spaces.
template <typename T>
__device__ __forceinline__ void suffix(T *out, const T *world, const T *local, int changed, T value) {
  T W[16], next[16], joint[16];
  for (int k = 0; k < 16; ++k) W[k] = changed ? world[16*(changed-1)+k] : T(k%5 == 0);
  for (int j = changed; j < N; ++j) {
    const T *transform;
    if (j == changed) {
      if (!model::joint_transform_scalar<T>(j, value, joint)) asm("trap;");
      transform = joint;
    } else {
      transform = local + 16*j;
    }
    glass::thread::gemm<T, 4, 4, 4>(T(1), W, transform, next);
    for (int k = 0; k < 16; ++k) W[k] = next[k];
  }
  for (int k = 0; k < 16; ++k) out[k] = W[k];
}

// Two warps, each over its own anchor set; two repeats of the same work so a
// nondeterministic result shows without any oracle.
template <typename T>
__global__ void probe(const T *anchors_in, const T *q_in, T *output) {
  __shared__ T anchors[2*32*N];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  T *a = anchors + warp*32*N;
  for (int repeat = 0; repeat < 2; ++repeat) {
    if (lane == 0) for (int k = 0; k < 32*N; ++k) a[k] = anchors_in[warp*32*N + k];
    __syncwarp();
    if (lane < N) {
      T out[16];
      suffix<T>(out, a, a + 16*N, lane, q_in[warp*N + lane]);
      for (int k = 0; k < 16; ++k) output[(repeat*64 + threadIdx.x)*16 + k] = out[k];
    }
    __syncwarp();
  }
}

static void ck(cudaError_t e) {
  if (e != cudaSuccess) { fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); exit(2); }
}

template <typename T>
static int run(const char *path) {
  const size_t n_anchor = 2*32*N, n_q = 2*N, n_out = 2*64*16;
  std::vector<float> raw(n_anchor + n_q);
  FILE *f = fopen(path, "rb");
  if (!f || fread(raw.data(), sizeof(float), raw.size(), f) != raw.size()) {
    fprintf(stderr, "cannot read %zu floats from %s\n", raw.size(), path); return 3;
  }
  fclose(f);
  std::vector<T> anchors(n_anchor), q(n_q), out(n_out, T(789));
  for (size_t i = 0; i < n_anchor; ++i) anchors[i] = static_cast<T>(raw[i]);
  for (size_t i = 0; i < n_q; ++i) q[i] = static_cast<T>(raw[n_anchor + i]);
  T *d_a, *d_q, *d_out;
  ck(cudaMalloc(&d_a, n_anchor*sizeof(T))); ck(cudaMalloc(&d_q, n_q*sizeof(T)));
  ck(cudaMalloc(&d_out, n_out*sizeof(T)));
  ck(cudaMemcpy(d_a, anchors.data(), n_anchor*sizeof(T), cudaMemcpyHostToDevice));
  ck(cudaMemcpy(d_q, q.data(), n_q*sizeof(T), cudaMemcpyHostToDevice));
  ck(cudaMemcpy(d_out, out.data(), n_out*sizeof(T), cudaMemcpyHostToDevice));
  probe<T><<<1, 64>>>(d_a, d_q, d_out);
  ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
  ck(cudaMemcpy(out.data(), d_out, n_out*sizeof(T), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n_out; ++i) printf("%.17g\n", static_cast<double>(out[i]));
  cudaFree(d_a); cudaFree(d_q); cudaFree(d_out);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s <f32|f64> <in.bin>\n", argv[0]); return 1; }
  if (!strcmp(argv[1], "f32")) return run<float>(argv[2]);
  if (!strcmp(argv[1], "f64")) return run<double>(argv[2]);
  fprintf(stderr, "unknown dtype %s\n", argv[1]); return 1;
}
