/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Detection module, containing codes implementing the
 * face detection method described in the following paper:
 *
 *
 *   Funnel-structured cascade for multi-view face detection with alignment awareness,
 *   Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
 *   In Neurocomputing (under review)
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Shuzhe Wu (a Ph.D supervised by Prof. Shiguang Shan)
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

#include "feat/lab_feature_map.h"

#include <cmath>

#include "util/math_func.h"

namespace seeta {
namespace fd {

void LABFeatureMap::Compute(const uint8_t* input, int32_t width,
    int32_t height) {
  if (input == nullptr || width <= 0 || height <= 0) {
    return;  // @todo handle the errors!!!
  }

  Reshape(width, height);
  ComputeIntegralImages(input);
  ComputeRectSum();
  ComputeFeatureMap();
}

float LABFeatureMap::GetStdDev() const {
  double mean;
  double m2;
  double area = roi_.width * roi_.height;

  int32_t top_left;
  int32_t top_right;
  int32_t bottom_left;
  int32_t bottom_right;

  if (roi_.x != 0) {
    if (roi_.y != 0) {
      top_left = (roi_.y - 1) * width_ + roi_.x - 1;
      top_right = top_left + roi_.width;
      bottom_left = top_left + roi_.height * width_;
      bottom_right = bottom_left + roi_.width;

      mean = (int_img_[bottom_right] - int_img_[bottom_left] +
        int_img_[top_left] - int_img_[top_right]) / area;
      m2 = (square_int_img_[bottom_right] - square_int_img_[bottom_left] +
        square_int_img_[top_left] - square_int_img_[top_right]) / area;
    } else {
      bottom_left = (roi_.height - 1) * width_ + roi_.x - 1;
      bottom_right = bottom_left + roi_.width;

      mean = (int_img_[bottom_right] - int_img_[bottom_left]) / area;
      m2 = (square_int_img_[bottom_right] - square_int_img_[bottom_left]) / area;
    }
  } else {
    if (roi_.y != 0) {
      top_right = (roi_.y - 1) * width_ + roi_.width - 1;
      bottom_right = top_right + roi_.height * width_;

      mean = (int_img_[bottom_right] - int_img_[top_right]) / area;
      m2 = (square_int_img_[bottom_right] - square_int_img_[top_right]) / area;
    } else {
      bottom_right = (roi_.height - 1) * width_ + roi_.width - 1;
      mean = int_img_[bottom_right] / area;
      m2 = square_int_img_[bottom_right] / area;
    }
  }

  return static_cast<float>(std::sqrt(m2 - mean * mean));
}

void LABFeatureMap::Reshape(int32_t width, int32_t height) {
  width_ = width;
  height_ = height;

  int32_t len = width_ * height_;
  feat_map_.resize(len);
  rect_sum_.resize(len);
  int_img_.resize(len);
  square_int_img_.resize(len);
}

void LABFeatureMap::ComputeIntegralImages(const uint8_t* input) {
  int32_t len = width_ * height_;

  seeta::fd::MathFunction::UInt8ToInt32(input, int_img_.data(), len);
  seeta::fd::MathFunction::Square(int_img_.data(), square_int_img_.data(), len);
  Integral(int_img_.data());
  Integral(square_int_img_.data());
}

void LABFeatureMap::ComputeRectSum() {
  int32_t width = width_ - rect_width_;
  int32_t height = height_ - rect_height_;
  const int32_t* int_img = int_img_.data();
  int32_t* rect_sum = rect_sum_.data();

  *rect_sum = *(int_img + (rect_height_ - 1) * width_ + rect_width_ - 1);
  seeta::fd::MathFunction::VectorSub(int_img + (rect_height_ - 1) * width_ +
    rect_width_, int_img + (rect_height_ - 1) * width_, rect_sum + 1, width);

#pragma omp parallel num_threads(SEETA_NUM_THREADS)
  {
#pragma omp for nowait
    for (int32_t i = 1; i <= height; i++) {
      const int32_t* top_left = int_img + (i - 1) * width_;
      const int32_t* top_right = top_left + rect_width_ - 1;
      const int32_t* bottom_left = top_left + rect_height_ * width_;
      const int32_t* bottom_right = bottom_left + rect_width_ - 1;
      int32_t* dest = rect_sum + i * width_;

      *(dest++) = (*bottom_right) - (*top_right);
      seeta::fd::MathFunction::VectorSub(bottom_right + 1, top_right + 1, dest, width);
      seeta::fd::MathFunction::VectorSub(dest, bottom_left, dest, width);
      seeta::fd::MathFunction::VectorAdd(dest, top_left, dest, width);
    }
  }
}

void LABFeatureMap::ComputeFeatureMap() {
  int32_t width = width_ - rect_width_ * num_rect_;
  int32_t height = height_ - rect_height_ * num_rect_;
  int32_t offset = width_ * rect_height_;
  uint8_t* feat_map = feat_map_.data();

#pragma omp parallel num_threads(SEETA_NUM_THREADS)
  {
#pragma omp for nowait
    for (int32_t r = 0; r <= height; r++) {
      for (int32_t c = 0; c <= width; c++) {
        uint8_t* dest = feat_map + r * width_ + c;
        *dest = 0;

        int32_t white_rect_sum = rect_sum_[(r + rect_height_) * width_ + c + rect_width_];
        int32_t black_rect_idx = r * width_ + c;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x80 : 0x0);
        black_rect_idx += rect_width_;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x40 : 0x0);
        black_rect_idx += rect_width_;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x20 : 0x0);
        black_rect_idx += offset;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x08 : 0x0);
        black_rect_idx += offset;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x01 : 0x0);
        black_rect_idx -= rect_width_;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x02 : 0x0);
        black_rect_idx -= rect_width_;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x04 : 0x0);
        black_rect_idx -= offset;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x10 : 0x0);
      }
    }
  }
}

}  // namespace fd
}  // namespace seeta
