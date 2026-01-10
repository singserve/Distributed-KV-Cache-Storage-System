cmake_minimum_required(VERSION 3.18)
project(nvlink_transfer LANGUAGES C CXX CUDA)

# Build shared library
add_library(nvlink_transfer SHARED nvlink_transfer.cu)
set_target_properties(nvlink_transfer PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# On Windows, ensure the CUDA runtime is linked
find_package(CUDA REQUIRED)

# Use C++14 compatibility
target_compile_features(nvlink_transfer PRIVATE cxx_std_14)

# Link CUDA runtime if needed (usually automatic)
if (TARGET CUDA::cudart)
    target_link_libraries(nvlink_transfer PRIVATE CUDA::cudart)
else()
    target_link_libraries(nvlink_transfer PRIVATE cudart)
endif()

# Installation helper (optional)
install(TARGETS nvlink_transfer DESTINATION lib)
install(FILES nvlink_transfer.cu DESTINATION include)

message(STATUS "nvlink_transfer library configured")
