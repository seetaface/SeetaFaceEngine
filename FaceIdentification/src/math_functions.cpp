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
#include <xmmintrin.h>
#include <cstdint>

#ifdef __ARM_EABI__
#include "arm_neon.h"
#else
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#ifdef __ARM_EABI__

#ifdef __ARM_NEON__

float simd_dot(const float* src1, const float* src2, const long& count) {
	long i = 0;

	float32x4_t sum_vec = vdupq_n_f32(0);
	float32x4_t data_a, data_b;
	for (; i <count-3; i+=4){
		data_a = vld1q_f32(&src1[i]);
		data_b = vld1q_f32(&src2[i]);
		sum_vec = vaddq_f32(sum_vec, vmulq_f32(data_a, data_b));
	}

	float sum = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3];
	for (; i < count; i++){
		sum += src1[i] * src2[i];
	}
	return sum;
}

#else // __ARM_NEON__

float simd_dot(const float* x, const float* y, const long& len) {
	float inner_prod = 0.0f;
	for (long i = 0; i < len; i++) {
		inner_prod += x[i] * y[i];
	}
	return inner_prod;
}

#endif // end __ARM_NEON__

#else // __ARM_EABI__

float simd_dot(const float* x, const float* y, const long& len) {
  float inner_prod = 0.0f;
  __m128 X, Y; // 128-bit values
  __m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
  float temp[4];

  long i;
  for (i = 0; i + 4 < len; i += 4) {
      X = _mm_loadu_ps(x + i); // load chunk of 4 floats
      Y = _mm_loadu_ps(y + i);
      acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
  }
  _mm_storeu_ps(&temp[0], acc); // store acc into an array
  inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

  // add the remaining values
  for (; i < len; ++i) {
      inner_prod += x[i] * y[i];
  }
  return inner_prod;
}

#endif // end __ARM_EABI__

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
