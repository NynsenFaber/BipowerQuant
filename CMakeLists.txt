cmake_minimum_required(VERSION 3.18)
project(BipowerQuant LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimised build. The engine exists to be fast, and a bare
# `cmake ..` with no build type otherwise produces an unoptimised module on the
# single-config generators (Make, Ninja) — a silent order-of-magnitude slowdown.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Resolve the interpreter with the modern FindPython module rather than letting
# pybind11 fall back to the removed FindPythonLibs. `Development.Module` asks for
# only what an extension module needs to link, which is the component that
# actually exists inside a virtualenv and on the CI runner images.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v3.0.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(bipower_core src/bindings.cpp)
target_include_directories(bipower_core PRIVATE src)

if(MSVC)
    target_compile_options(bipower_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(bipower_core PRIVATE -Wall -Wextra)
endif()

# Drop the module next to the Python sources that import it, for every generator.
# Multi-config generators (MSBuild, Xcode) append the configuration name to the
# output directory unless the per-config variable is set too, which is why both
# forms are needed to land the file in one predictable place on all three OSes.
set(BIPOWER_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python")
set_target_properties(bipower_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${BIPOWER_MODULE_DIR}")
foreach(config ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${config} config_upper)
    set_target_properties(bipower_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${BIPOWER_MODULE_DIR}")
endforeach()
