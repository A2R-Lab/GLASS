cmake_minimum_required(VERSION 3.23)
project(glass LANGUAGES CXX)

# GLASS is header-only: this file exists so CMake consumers get a proper
# INTERFACE target via add_subdirectory / FetchContent / find_package.
#
#   FetchContent_Declare(glass GIT_REPOSITORY https://github.com/A2R-Lab/GLASS.git)
#   FetchContent_MakeAvailable(glass)
#   target_link_libraries(my_kernel PRIVATE glass::glass)
#
# Options:
#   GLASS_ENABLE_MATHDX  — also expose the MathDx include dir (cuBLASDx /
#                          cuSOLVERDx) for the glass::nvidia:: interface.
#                          Requires MATHDX_ROOT (env or cache variable).
#   GLASS_BUILD_EXAMPLES — build examples/ (requires the CUDA language).

option(GLASS_ENABLE_MATHDX "Expose MathDx includes for glass::nvidia::" OFF)
option(GLASS_BUILD_EXAMPLES "Build the examples" OFF)

add_library(glass INTERFACE)
add_library(glass::glass ALIAS glass)

target_include_directories(glass INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/glass>)
target_compile_features(glass INTERFACE cxx_std_17)

if(GLASS_ENABLE_MATHDX)
    if(NOT DEFINED MATHDX_ROOT AND DEFINED ENV{MATHDX_ROOT})
        set(MATHDX_ROOT $ENV{MATHDX_ROOT})
    endif()
    if(NOT DEFINED MATHDX_ROOT)
        message(FATAL_ERROR "GLASS_ENABLE_MATHDX requires MATHDX_ROOT (cmake -DMATHDX_ROOT=... or the env var)")
    endif()
    target_include_directories(glass INTERFACE
        ${MATHDX_ROOT}/include
        ${MATHDX_ROOT}/external/cutlass/include)
endif()

if(GLASS_BUILD_EXAMPLES)
    enable_language(CUDA)
    add_subdirectory(examples)
endif()

# ── install / find_package(glass) ─────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS glass EXPORT glassTargets)
install(FILES glass.cuh glass-cgrps.cuh glass-nvidia.cuh glass-defaults.cuh
              glass-dispatch.cuh
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/glass)
install(DIRECTORY src DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/glass
        FILES_MATCHING PATTERN "*.cuh")
install(EXPORT glassTargets NAMESPACE glass::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/glass)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/glassConfig.cmake.in
     "@PACKAGE_INIT@\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/glassTargets.cmake\")\n")
configure_package_config_file(
    ${CMAKE_CURRENT_BINARY_DIR}/glassConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/glassConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/glass)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/glassConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/glass)
