/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Identification module, containing codes implementing the
 * face identification method described in the following paper:
 *
 *   
 *   VIPLFaceNet: An Open Source Deep Face Recognition SDK,
 *   Xin Liu, Meina Kan, Wanglong Wu, Shiguang Shan, Xilin Chen.
 *   In Frontiers of Computer Science.
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Zining Xu(a M.S. supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems. 
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include "math_functions.h"
#include <cstdint>

#if defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
     #include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC-compatible compiler, targeting x86/x86-64 */
     #include <x86intrin.h>
#elif defined(__GNUC__) && defined(__ARM_NEON__)
/* GCC-compatible compiler, targeting ARM with NEON */
     #include <arm_neon.h>
#endif



#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__) || defined(_MSC_VER) )
float simd_dot(const float* x, const float* y, const long& len) {
 #pragma message("USE SSE")
  float inner_prod = 0.0f;
  __m128 X,Y,Z; // 128-bit values
  __m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
  float temp[4];

  long i;
  for (i = 0; i + 4 < len; i += 4) {
      X = _mm_loadu_ps(x + i); // load chunk of 4 floats
      Y = _mm_loadu_ps(y + i);
      Z = _mm_mul_ps(X, Y);
      acc = _mm_add_ps(acc, Z);
  }
  _mm_storeu_ps(&temp[0], acc); // store acc into an array
  inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

  // add the remaining values
  for (; i < len; ++i) {
      inner_prod += x[i] * y[i];
  }
  return inner_prod;
}
#else
float simd_dot(const float* x, const float* y, const long& len) {
#pragma message("USE NEON")
    float inner_prod=0.0f;
    float32x4_t X,Y,Z;// 128-bit values
    float32x4_t acc=vdupq_n_f32(0.0f);//set to (0, 0, 0, 0)
    long i;
    for (i = 0; i + 4 < len; i += 4) {
        X = vld1q_f32(x + i);// load chunk of 4 floats
        Y = vld1q_f32(y + i);
        Z = vmulq_f32(X, Y);
        acc = vaddq_f32(acc, Z);
    }
    inner_prod=vgetq_lane_f32(acc, 0)+vgetq_lane_f32(acc, 1)+vgetq_lane_f32(acc, 2) +vgetq_lane_f32(acc, 3);
    for (; i < len; ++i) {
        inner_prod += x[i] * y[i];
    }
    return inner_prod;
}
#endif


void matrix_procuct(const float* A, const float* B, float* C, const int n,
    const int m, const int k, bool ta, bool tb) {
#ifdef _BLAS
  arma::fmat mA = ta ? arma::fmat(A, k, n).t() : arma::fmat(A, n, k);
  arma::fmat mB = tb ? arma::fmat(B, m, k).t() : arma::fmat(B, k, m);
  arma::fmat mC(C, n, m, false);
  mC = mA * mB;
#else
  CHECK_TRUE(ta && !tb);
  const float* x = B;
  for (int i = 0, idx = 0; i < m; ++i) {
    const float* y = A;
    for (int j = 0; j < n; ++j, ++idx) {
      C[idx] = simd_dot(x, y, k);
      y += k;
    }
    x += k;
  }
#endif
}
