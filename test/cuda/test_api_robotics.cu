#include "../../glass.cuh"
#include <cstdio>

// Compile-only overload canary; the robotics pytest suite supplies numerical
// NumPy/SciPy/Pinocchio and finite-difference oracles.
__global__ void compile_robotics_contracts(bool run, float* p) {
    if (!run) return;

    glass::block::svd3(p, p, p, p);
    glass::block::closest_rotation(p, p);
    glass::warp::eig3(p, p, p);
    glass::warp::svd3(p, p, p, p);
    glass::warp::closest_rotation(p, p);
    glass::thread::eig3(p, p, p);
    glass::thread::svd3(p, p, p, p);
    (void)glass::block::sphere_sphere_dist(p, 1.0f, p, 1.0f);
    (void)glass::block::sphere_box_dist(p, 1.0f, p);

    glass::block::quat_conj(p, p);
    glass::block::quat_exp(p, p);
    glass::block::quat_log(p, p);
    glass::block::quat_rotate(p, p, p);
    glass::block::quat_to_basis(p, p, p, p);
    glass::block::quat_retract(p, p, p);
    glass::warp::quat_conj(p, p);
    glass::warp::quat_exp(p, p);
    glass::warp::quat_log(p, p);
    glass::warp::quat_rotate(p, p, p);
    glass::warp::quat_to_basis(p, p, p, p);
    glass::warp::quat_retract(p, p, p);
    glass::thread::quat_conj(p, p);
    glass::thread::quat_exp(p, p);
    glass::thread::quat_log(p, p);
    glass::thread::quat_rotate(p, p, p);
    glass::thread::quat_to_basis(p, p, p, p);

    glass::block::se3_Q_block(p, p, p);
    glass::block::se3_retract(p, p, p, p);
    glass::block::se3_difference(p, p, p, p);
    glass::block::se3_retract_jacobian_q(p, p, p);
    glass::block::se3_retract_jacobian_v(p, p, p);
    glass::warp::se3_Q_block(p, p, p);
    glass::warp::se3_retract(p, p, p, p);
    glass::warp::se3_difference(p, p, p, p);
    glass::warp::se3_retract_jacobian_q(p, p, p);
    glass::warp::se3_retract_jacobian_v(p, p, p);
    glass::thread::se3_Q_block(p, p, p);
    glass::thread::se3_difference(p, p, p, p);
    glass::thread::se3_retract_jacobian_q(p, p, p);
    glass::thread::se3_retract_jacobian_v(p, p, p);

    glass::block::skew(p, p);
    glass::block::so3_exp(p, p);
    glass::block::so3_log(p, p);
    glass::block::so3_right_jacobian(p, p);
    glass::block::so3_right_jacobian_inv(p, p);
    glass::block::so3_left_jacobian(p, p);
    glass::block::so3_left_jacobian_inv(p, p);
    glass::warp::skew(p, p);
    glass::warp::so3_exp(p, p);
    glass::warp::so3_log(p, p);
    glass::warp::so3_right_jacobian(p, p);
    glass::warp::so3_right_jacobian_inv(p, p);
    glass::warp::so3_left_jacobian(p, p);
    glass::warp::so3_left_jacobian_inv(p, p);
    glass::thread::skew(p, p);
    glass::thread::so3_exp(p, p);
    glass::thread::so3_log(p, p);
    glass::thread::so3_right_jacobian(p, p);
    glass::thread::so3_right_jacobian_inv(p, p);
    glass::thread::so3_left_jacobian(p, p);
    glass::thread::so3_left_jacobian_inv(p, p);

    glass::block::soc_project(p, p, 3);
    glass::warp::soc_project(p, p, 3);
    glass::block::force_cross_dual(p, p);
    glass::warp::motion_cross(p, p);
    glass::warp::force_cross(p, p);
    glass::warp::force_cross_dual(p, p);
    glass::thread::motion_cross(p, p);
    glass::thread::force_cross(p, p);
    glass::thread::force_cross_dual(p, p);
    glass::block::spatial_inertia(p, p);
    glass::warp::spatial_inertia(p, p);
    glass::thread::spatial_inertia(p, p);
    glass::block::motion_transform(p, p, p);
    glass::block::force_transform(p, p, p);
    glass::block::motion_transform_mul(1.0f, p, p, p, 1.0f, p);
    glass::block::force_transform_mul(1.0f, p, p, p, 1.0f, p);
    glass::warp::motion_transform(p, p, p);
    glass::warp::force_transform(p, p, p);
    glass::warp::motion_transform_mul(1.0f, p, p, p, 1.0f, p);
    glass::warp::force_transform_mul(1.0f, p, p, p, 1.0f, p);
    glass::thread::motion_transform(p, p, p);
    glass::thread::force_transform(p, p, p);
    glass::thread::motion_transform_mul(1.0f, p, p, p, 1.0f, p);
    glass::thread::force_transform_mul(1.0f, p, p, p, 1.0f, p);
}

int main() {
    std::puts("1");
    return 0;
}
