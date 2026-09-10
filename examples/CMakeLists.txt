# Built via the root CMakeLists with -DGLASS_BUILD_EXAMPLES=ON.
# Example 06 (glass::nvidia::) additionally needs -DGLASS_ENABLE_MATHDX=ON.

file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS "[0-9][0-9]_*.cu")

foreach(src ${EXAMPLE_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    if(name STREQUAL "05_nvidia_gemm" AND NOT GLASS_ENABLE_MATHDX)
        continue()
    endif()
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE glass::glass)
    set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES native)
    if(name STREQUAL "05_nvidia_gemm")
        target_compile_definitions(${name} PRIVATE GLASS_BENCH_CUBLASDX)
    endif()
endforeach()
