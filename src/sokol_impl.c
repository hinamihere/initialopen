// Implementation unit for the header-only sokol libraries (backend: SOKOL_GLCORE, set in CMake).
#ifndef NDEBUG
#define SOKOL_DEBUG  // validation layer and full error messages in debug builds
#endif
#define SOKOL_IMPL
#include "sokol_gfx.h"
#include "sokol_log.h"
