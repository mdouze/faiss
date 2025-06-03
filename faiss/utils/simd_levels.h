/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

namespace faiss {

/* SIMD levels, used as template parameters. All hardcoded, single enabled SIMD
level determined from the compiler flags #ifdef __x86_64__ #ifdef __AVX512F__
#define COMPILE_SIMD_AVX512F
#ifdef __AVX2__
#define COMPILE_SIMD_AVX2
#endif
#elif defined(__aarch64__)
#define COMPILE_SIMD_NEON
#endif
#endif

// levels are defined, even for
 * architectures different of the current one. */
enum class SIMDLevel {
    NONE,
    // x86
    AVX2,
    AVX512F,
    // arm
    ARM_NEON,
    ARM_SVE,
    // ppc
    PPC_ALTIVEC,
};

/* Current SIMD configuration. This static class manages the current SIMD level
 * and intializes it from the cpuid and the FAISS_SIMD_LEVEL
 * environment variable  */
struct SIMDConfig {
    static SIMDLevel level;
    static void set_level(SIMDLevel level);
    static SIMDLevel get_level();
    static std::string get_level_name();
    static const char* level_names[];

    SIMDConfig();
};

// dummy dispatching function that calls the hardcoded SIMD level

#ifdef COMPILE_SIMD_AVX2
#define DISPATCH_SIMDLevel(f, ...) return f<SIMDLevel::AVX2>(__VA_ARGS__)
#elif defined(COMPILE_SIMD_AVX512F)
#define DISPATCH_SIMDLevel(f, ...) return f<SIMDLevel::AVX512F>(__VA_ARGS__)
#elif defined(COMPILE_SIMD_NEON)
#define DISPATCH_SIMDLevel(f, ...) return f<SIMDLevel::ARM_NEON>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel(f, ...) return f<SIMDLevel::NONE>(__VA_ARGS__)
#endif

} // namespace faiss
