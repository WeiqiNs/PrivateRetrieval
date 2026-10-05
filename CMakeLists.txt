cmake_minimum_required(VERSION 3.25)

project(
        PrivateRetrieval
        VERSION 0.1
        DESCRIPTION "Private embedding retrieval with inner-product functional encryption over LibPFE."
        LANGUAGES CXX
)

set(CMAKE_CXX_EXTENSIONS OFF)
if (PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif ()

option(PRIVRET_BUILD_TESTS "Build the PrivateRetrieval test suite" ${PROJECT_IS_TOP_LEVEL})

set(RBP_CURVES bls12_381 CACHE STRING "Curves to build, from cmake/RBPCurves.cmake")

include(FetchContent)
FetchContent_Declare(PFE
        GIT_REPOSITORY https://github.com/WeiqiNs/LibPFE.git
        GIT_TAG main
        GIT_SHALLOW TRUE
        FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(PFE)

if (NOT TARGET RBP::BLS12_381)
    message(FATAL_ERROR "PrivateRetrieval needs LibRBP built with the bls12_381 curve (RBP_CURVES).")
endif ()

add_library(privret_core INTERFACE)
target_include_directories(privret_core INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(privret_core INTERFACE PFE::PFE)
target_compile_features(privret_core INTERFACE cxx_std_20)

add_executable(privret src/privret.cpp)
target_link_libraries(privret PRIVATE privret_core)

if (PRIVRET_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()
