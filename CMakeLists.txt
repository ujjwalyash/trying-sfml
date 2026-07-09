cmake_minimum_required(VERSION 3.28)

# set(CMAKE_CXX_COMPILER "g++") 
set(CMAKE_CXX_COMPILER "clang++") 

project(CMakeSFMLProject LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.1.0
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM)
FetchContent_MakeAvailable(SFML)

# target_compile_features(main PRIVATE cxx_std_26)
# Enable modern C++ standards (C++20 or C++23)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(main 
                src/main.cpp
                src/particle.cpp
                src/spring.cpp
                src/creature.cpp
                src/muscle.cpp
                src/environment.cpp
                src/collision_handling.cpp

                src/cuda.cu
)

# Turn on strict compiler warnings (The "Teacher Mode")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
target_compile_options(main PRIVATE 
#     -Wall          # Turns on all basic warnings
#     -Wextra        # Turns on extra useful warnings
#     -Wpedantic     # Forces strict adherence to the C++ standard
#     -Wshadow       # Warns if you accidentally reuse a variable name
    $<$<COMPILE_LANGUAGE:CXX>: -Wshadow -Wpedantic -Wextra -Wall>
    $<$<COMPILE_LANGUAGE:CUDA>: -Xptxas="-v">
    
    # $<$<COMPILE_LANGUAGE:CXX>: -g>
    # $<$<COMPILE_LANGUAGE:CUDA>: -g -G>
    
    -O3
    -march=native
)
endif()

target_link_libraries(main PRIVATE SFML::Graphics)
target_include_directories(main SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/external")
        # -Wconversion   # Warns if data might get lost (e.g., assigning a float to an int)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT main)
