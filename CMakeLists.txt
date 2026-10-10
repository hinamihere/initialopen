cmake_minimum_required(VERSION 3.24)
project(initialopen LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

# Precompiled headers break with GCC when the build path has non-ASCII characters.
set(CMAKE_DISABLE_PRECOMPILE_HEADERS ON)

# --- SDL3: window, input, GL context, audio ------------------------------
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
FetchContent_Declare(SDL3
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG release-3.2.10
  GIT_SHALLOW TRUE)

# --- Jolt Physics: rigid bodies + WheeledVehicleController ---------------
set(TARGET_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(TARGET_HELLO_WORLD OFF CACHE BOOL "" FORCE)
set(TARGET_PERFORMANCE_TEST OFF CACHE BOOL "" FORCE)
set(TARGET_SAMPLES OFF CACHE BOOL "" FORCE)
set(TARGET_VIEWER OFF CACHE BOOL "" FORCE)
set(ENABLE_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
set(USE_STATIC_MSVC_RUNTIME_LIBRARY OFF CACHE BOOL "" FORCE)
FetchContent_Declare(JoltPhysics
  GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
  GIT_TAG v5.3.0
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR Build)

# --- Header-only libraries -----------------------------------------------
# sokol and stb have no release tags; pinned to known-good commits.
FetchContent_Declare(sokol
  GIT_REPOSITORY https://github.com/floooh/sokol.git
  GIT_TAG 401f21f8b7039258c35fef75c11d9a8a0e616771)
FetchContent_Declare(cgltf
  GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
  GIT_TAG v1.14
  GIT_SHALLOW TRUE)
FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG 2c980bb59875b0d32144a71867fbdebb2f77cd20)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
  GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(SDL3 JoltPhysics json sokol cgltf stb)

add_executable(initialopen
  src/main.cpp
  src/sokol_impl.c
  src/assets.cpp
  src/physics.cpp
  src/renderer.cpp
  src/audio.cpp
  src/timing.cpp)
target_include_directories(initialopen PRIVATE src ${sokol_SOURCE_DIR} ${cgltf_SOURCE_DIR} ${stb_SOURCE_DIR})
target_compile_definitions(initialopen PRIVATE SOKOL_GLCORE)
target_link_libraries(initialopen PRIVATE SDL3::SDL3-static Jolt nlohmann_json::nlohmann_json)
if(WIN32)
  target_link_libraries(initialopen PRIVATE opengl32)
  if(MINGW)
    target_link_options(initialopen PRIVATE -static -static-libgcc -static-libstdc++ $<$<CONFIG:Release>:-s>)
  endif()
endif()
