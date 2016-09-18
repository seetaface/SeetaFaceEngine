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

#include <cmath>
#include "feat/surf_feature_map.h"

namespace seeta {
namespace fd {

void SURFFeaturePool::Create() {
  if (sample_height_ - patch_min_height_ <= sample_width_ - patch_min_width_) {
    for (size_t i = 0; i < format_.size(); i++) {
      const SURFPatchFormat & format = format_[i];
      for (int32_t h = patch_min_height_; h <= sample_height_;
          h += patch_size_inc_step_) {
        if (h % format.num_cell_per_col != 0 || h % format.height != 0)
          continue;
        int32_t w = h / format.height * format.width;
        if (w % format.num_cell_per_row != 0 || w < patch_min_width_ ||
            w > sample_width_)
          continue;
        AddAllFeaturesToPool(w, h, format.num_cell_per_row,
          format.num_cell_per_col);
      }
    }
  } else {
    for (size_t i = 0; i < format_.size(); i++) {
      const SURFPatchFormat & format = format_[i];
      for (int32_t w = patch_min_width_; w <= patch_min_width_;
          w += patch_size_inc_step_) {
        if (w % format.num_cell_per_row != 0 || w % format.width != 0)
          continue;
        int32_t h = w / format.width * format.height;
        if (h % format.num_cell_per_col != 0 || h < patch_min_height_ ||
            h > sample_height_)
          continue;
        AddAllFeaturesToPool(w, h, format.num_cell_per_row,
          format.num_cell_per_col);
      }
    }
  }
}

void SURFFeaturePool::AddPatchFormat(int32_t width, int32_t height,
    int32_t num_cell_per_row, int32_t num_cell_per_col) {
  for (size_t i = 0; i < format_.size(); i++) {
    const SURFPatchFormat & format = format_[i];
    if (format.height == height &&
      format.width == width &&
      format.num_cell_per_row == num_cell_per_row &&
      format.num_cell_per_col == num_cell_per_col)
      return;
  }

  SURFPatchFormat new_format;
  new_format.height = height;
  new_format.width = width;
  new_format.num_cell_per_row = num_cell_per_row;
  new_format.num_cell_per_col = num_cell_per_col;
  format_.push_back(new_format);
}

void SURFFeaturePool::AddAllFeaturesToPool(int32_t width, int32_t height,
    int32_t num_cell_per_row, int32_t num_cell_per_col) {
  SURFFeature feat;
  feat.patch.width = width;
  feat.patch.height = height;
  feat.num_cell_per_row = num_cell_per_row;
  feat.num_cell_per_col = num_cell_per_col;

  for (int32_t y = 0; y <= sample_height_ - height; y += patch_move_step_y_) {
    feat.patch.y = y;
    for (int32_t x = 0; x <= sample_width_ - width; x += patch_move_step_x_) {
      feat.patch.x = x;
      pool_.push_back(feat);
    }
  }
}

void SURFFeatureMap::Compute(const uint8_t* input, int32_t width,
    int32_t height) {
  if (input == nullptr || width <= 0 || height <= 0) {
    return;  // @todo handle the error!
  }
  Reshape(width, height);
  ComputeGradientImages(input);
  ComputeIntegralImages();
}

void SURFFeatureMap::GetFeatureVector(int32_t feat_id, float* feat_vec) {
  if (buf_valid_[feat_id] == 0) {
    ComputeFeatureVector(feat_pool_[feat_id], feat_vec_buf_[feat_id].data());
    NormalizeFeatureVectorL2(feat_vec_buf_[feat_id].data(),
      feat_vec_normed_buf_[feat_id].data(),
      static_cast<int32_t>(feat_vec_normed_buf_[feat_id].size()));
    buf_valid_[feat_id] = 1;
    buf_valid_reset_ = true;
  }

  std::memcpy(feat_vec, feat_vec_normed_buf_[feat_id].data(),
    feat_vec_normed_buf_[feat_id].size() * sizeof(float));
}

void SURFFeatureMap::InitFeaturePool() {
  feat_pool_.AddPatchFormat(1, 1, 2, 2);
  feat_pool_.AddPatchFormat(1, 2, 2, 2);
  feat_pool_.AddPatchFormat(2, 1, 2, 2);
  feat_pool_.AddPatchFormat(2, 3, 2, 2);
  feat_pool_.AddPatchFormat(3, 2, 2, 2);
  feat_pool_.Create();

  int32_t feat_pool_size = static_cast<int32_t>(feat_pool_.size());
  feat_vec_buf_.resize(feat_pool_size);
  feat_vec_normed_buf_.resize(feat_pool_size);
  for (size_t i = 0; i < feat_pool_size; i++) {
    int32_t dim = GetFeatureVectorDim(static_cast<int32_t>(i));
    feat_vec_buf_[i].resize(dim);
    feat_vec_normed_buf_[i].resize(dim);
  }
  buf_valid_.resize(feat_pool_size, 0);
}

void SURFFeatureMap::Reshape(int32_t width, int32_t height) {
  width_ = width;
  height_ = height;

  int32_t len = width_ * height_;
  grad_x_.resize(len);
  grad_y_.resize(len);
  int_img_.resize(len * kNumIntChannel);
  img_buf_.resize(len);
}

void SURFFeatureMap::ComputeGradientImages(const uint8_t* input) {
  int32_t len = width_ * height_;
  seeta::fd::MathFunction::UInt8ToInt32(input, img_buf_.data(), len);
  ComputeGradX(img_buf_.data());
  ComputeGradY(img_buf_.data());
}

void SURFFeatureMap::ComputeGradX(const int32_t* input) {
  int32_t* dx = grad_x_.data();
  int32_t len = width_ - 2;

#pragma omp parallel num_threads(SEETA_NUM_THREADS)
  {
#pragma omp for nowait
    for (int32_t r = 0; r < height_; r++) {
      const int32_t* src = input + r * width_;
      int32_t* dest = dx + r * width_;
      *dest = ((*(src + 1)) - (*src)) << 1;
      seeta::fd::MathFunction::VectorSub(src + 2, src, dest + 1, len);
      dest += (width_ - 1);
      src += (width_ - 1);
      *dest = ((*src) - (*(src - 1))) << 1;
    }
  }
}

void SURFFeatureMap::ComputeGradY(const int32_t* input) {
  int32_t* dy = grad_y_.data();
  int32_t len = width_;
  seeta::fd::MathFunction::VectorSub(input + width_, input, dy, len);
  seeta::fd::MathFunction::VectorAdd(dy, dy, dy, len);

#pragma omp parallel num_threads(SEETA_NUM_THREADS)
  {
#pragma omp for nowait
    for (int32_t r = 1; r < height_ - 1; r++) {
      const int32_t* src = input + (r - 1) * width_;
      int32_t* dest = dy + r * width_;
      seeta::fd::MathFunction::VectorSub(src + (width_ << 1), src, dest, len);
    }
  }
  int32_t offset = (height_ - 1) * width_;
  dy += offset;
  seeta::fd::MathFunction::VectorSub(input + offset, input + offset - width_,
    dy, len);
  seeta::fd::MathFunction::VectorAdd(dy, dy, dy, len);
}

void SURFFeatureMap::ComputeIntegralImages() {
  FillIntegralChannel(grad_x_.data(), 0);
  FillIntegralChannel(grad_y_.data(), 4);

  int32_t len = width_ * height_;
  seeta::fd::MathFunction::VectorAbs(grad_x_.data(), img_buf_.data(), len);
  FillIntegralChannel(img_buf_.data(), 1);
  seeta::fd::MathFunction::VectorAbs(grad_y_.data(), img_buf_.data(), len);
  FillIntegralChannel(img_buf_.data(), 5);
  MaskIntegralChannel();
  Integral();
}

void SURFFeatureMap::MaskIntegralChannel() {
  const int32_t* grad_x = grad_x_.data();
  const int32_t* grad_y = grad_y_.data();
  int32_t len = width_ * height_;
#ifdef USE_SSE
  __m128i dx;
  __m128i dy;
  __m128i dx_mask;
  __m128i dy_mask;
  __m128i zero = _mm_set1_epi32(0);
  __m128i xor_bits = _mm_set_epi32(0x0, 0x0, 0xffffffff, 0xffffffff);
  __m128i data;
  __m128i result;
  __m128i* src = reinterpret_cast<__m128i*>(int_img_.data());

  for (int32_t i = 0; i < len; i++) {
    dx = _mm_set1_epi32(*(grad_x++));
    dy = _mm_set1_epi32(*(grad_y++));
    dx_mask = _mm_xor_si128(_mm_cmplt_epi32(dx, zero), xor_bits);
    dy_mask = _mm_xor_si128(_mm_cmplt_epi32(dy, zero), xor_bits);

    data = _mm_loadu_si128(src);
    result = _mm_and_si128(data, dy_mask);
    _mm_storeu_si128(src++, result);
    data = _mm_loadu_si128(src);
    result = _mm_and_si128(data, dx_mask);
    _mm_storeu_si128(src++, result);
  }
#else
  int32_t dx, dy, dx_mask, dy_mask, cmp;
  int32_t xor_bits[] = {-1, -1, 0, 0};

  int32_t* src = int_img_.data();
  for (int32_t i = 0; i < len; i++) {
      dy = *(grad_y++);
      dx = *(grad_x++);
      
      cmp = dy < 0 ? 0xffffffff : 0x0;
      for (int32_t j = 0; j < 4; j++) {
          // cmp xor xor_bits
          dy_mask = cmp ^ xor_bits[j];
          *(src) = (*src) & dy_mask;
          src++;
      }
      
      cmp = dx < 0 ? 0xffffffff : 0x0;
      for (int32_t j = 0; j < 4; j++) {
          // cmp xor xor_bits
          dx_mask = cmp ^ xor_bits[j];
          *(src) = (*src) & dx_mask;
          src++;
      }
  }
#endif
}

void SURFFeatureMap::Integral() {
  int32_t* data = int_img_.data();
  int32_t len = kNumIntChannel * width_;

  // Cummulative sum by row
  for (int32_t r = 0; r < height_ - 1; r++) {
    int32_t* row1 = data + r * len;
    int32_t* row2 = row1 + len;
    seeta::fd::MathFunction::VectorAdd(row1, row2, row2, len);
  }
  // Cummulative sum by column
  for (int32_t r = 0; r < height_; r++)
    VectorCumAdd(data + r * len, len, kNumIntChannel);
}

void SURFFeatureMap::VectorCumAdd(int32_t* x, int32_t len,
    int32_t num_channel) {
#ifdef USE_SSE
  __m128i x1;
  __m128i y1;
  __m128i z1;
  __m128i* x2 = reinterpret_cast<__m128i*>(x);
  __m128i* y2 = reinterpret_cast<__m128i*>(x + num_channel);
  __m128i* z2 = y2;

  len = len / num_channel - 1;
  for (int32_t i = 0; i < len; i++) {
    // first 4 channels
    x1 = _mm_loadu_si128(x2++);
    y1 = _mm_loadu_si128(y2++);
    z1 = _mm_add_epi32(x1, y1);
    _mm_storeu_si128(z2, z1);
    z2 = y2;

    // second 4 channels
    x1 = _mm_loadu_si128(x2++);
    y1 = _mm_loadu_si128(y2++);
    z1 = _mm_add_epi32(x1, y1);
    _mm_storeu_si128(z2, z1);
    z2 = y2;
  }
#else
  int32_t cols = len / num_channel - 1;
  for (int32_t i = 0; i < cols; i++) {
    int32_t* col1 = x + i * num_channel;
    int32_t* col2 = col1 + num_channel;
    seeta::fd::MathFunction::VectorAdd(col1, col2, col2, num_channel);
  }
#endif
}

void SURFFeatureMap::ComputeFeatureVector(const SURFFeature & feat,
    int32_t* feat_vec) {
  int32_t init_cell_x = roi_.x + feat.patch.x;
  int32_t init_cell_y = roi_.y + feat.patch.y;
  int32_t cell_width = feat.patch.width / feat.num_cell_per_row * kNumIntChannel;
  int32_t cell_height = feat.patch.height / feat.num_cell_per_col;
  int32_t row_width = width_ * kNumIntChannel;
  const int32_t* cell_top_left[kNumIntChannel];
  const int32_t* cell_top_right[kNumIntChannel];
  const int32_t* cell_bottom_left[kNumIntChannel];
  const int32_t* cell_bottom_right[kNumIntChannel];
  int* feat_val = feat_vec;
  const int32_t* int_img = int_img_.data();
  int32_t offset = 0;

  if (init_cell_y != 0) {
    if (init_cell_x != 0) {
      const int32_t* tmp_cell_top_right[kNumIntChannel];

      // cell #1
      offset = row_width * (init_cell_y - 1) +
        (init_cell_x - 1) * kNumIntChannel;
      for (int32_t i = 0; i < kNumIntChannel; i++) {
        cell_top_left[i] = int_img + (offset++);
        cell_top_right[i] = cell_top_left[i] + cell_width;
        cell_bottom_left[i] = cell_top_left[i] + row_width * cell_height;
        cell_bottom_right[i] = cell_bottom_left[i] + cell_width;
        *(feat_val++) = *(cell_bottom_right[i]) + *(cell_top_left[i]) -
                        *(cell_top_right[i]) - *(cell_bottom_left[i]);
        tmp_cell_top_right[i] = cell_bottom_right[i];
      }

      // cells in 1st row
      for (int32_t i = 1; i < feat.num_cell_per_row; i++) {
        for (int32_t j = 0; j < kNumIntChannel; j++) {
          cell_top_left[j] = cell_top_right[j];
          cell_top_right[j] += cell_width;
          cell_bottom_left[j] = cell_bottom_right[j];
          cell_bottom_right[j] += cell_width;
          *(feat_val++) = *(cell_bottom_right[j]) + *(cell_top_left[j]) -
                          *(cell_top_right[j]) - *(cell_bottom_left[j]);
        }
      }

      for (int32_t i = 0; i < kNumIntChannel; i++)
        cell_top_right[i] = tmp_cell_top_right[i];
    } else {
      const int32_t* tmp_cell_top_right[kNumIntChannel];

      // cell #1
      offset = row_width * (init_cell_y - 1) + cell_width - kNumIntChannel;
      for (int32_t i = 0; i < kNumIntChannel; i++) {
        cell_top_right[i] = int_img + (offset++);
        cell_bottom_right[i] = cell_top_right[i] + row_width * cell_height;
        tmp_cell_top_right[i] = cell_bottom_right[i];
        *(feat_val++) = *(cell_bottom_right[i]) - *(cell_top_right[i]);
      }

      // cells in 1st row
      for (int32_t i = 1; i < feat.num_cell_per_row; i++) {
        for (int32_t j = 0; j < kNumIntChannel; j++) {
          cell_top_left[j] = cell_top_right[j];
          cell_top_right[j] += cell_width;
          cell_bottom_left[j] = cell_bottom_right[j];
          cell_bottom_right[j] += cell_width;
          *(feat_val++) = *(cell_bottom_right[j]) + *(cell_top_left[j]) -
                          *(cell_top_right[j]) - *(cell_bottom_left[j]);
        }
      }

      for (int32_t i = 0; i < kNumIntChannel; i++)
        cell_top_right[i] = tmp_cell_top_right[i];
    }
  } else {
    if (init_cell_x != 0) {
      // cell #1
      offset = row_width * (cell_height - 1) +
        (init_cell_x - 1) * kNumIntChannel;
      for (int32_t i = 0; i < kNumIntChannel; i++) {
        cell_bottom_left[i] = int_img + (offset++);
        cell_bottom_right[i] = cell_bottom_left[i] + cell_width;
        *(feat_val++) = *(cell_bottom_right[i]) - *(cell_bottom_left[i]);
        cell_top_right[i] = cell_bottom_right[i];
      }

      // cells in 1st row
      for (int32_t i = 1; i < feat.num_cell_per_row; i++) {
        for (int32_t j = 0; j < kNumIntChannel; j++) {
          cell_bottom_left[j] = cell_bottom_right[j];
          cell_bottom_right[j] += cell_width;
          *(feat_val++) = *(cell_bottom_right[j]) - *(cell_bottom_left[j]);
        }
      }
    } else {
      // cell #1
      offset = row_width * (cell_height - 1) + cell_width - kNumIntChannel;
      for (int32_t i = 0; i < kNumIntChannel; i++) {
        cell_bottom_right[i] = int_img + (offset++);
        *(feat_val++) = *(cell_bottom_right[i]);
        cell_top_right[i] = cell_bottom_right[i];
      }

      // cells in 1st row
      for (int32_t i = 1; i < feat.num_cell_per_row; i++) {
        for (int32_t j = 0; j < kNumIntChannel; j++) {
          cell_bottom_left[j] = cell_bottom_right[j];
          cell_bottom_right[j] += cell_width;
          *(feat_val++) = *(cell_bottom_right[j]) - *(cell_bottom_left[j]);
        }
      }
    }
  }

  // from BR of last cell in current row to BR of first cell in next row
  offset = cell_height * row_width - feat.patch.width *
    kNumIntChannel + cell_width;

  // cells in following rows
  for (int32_t i = 1; i < feat.num_cell_per_row; i++) {
    // cells in 1st column
    if (init_cell_x == 0) {
      for (int32_t j = 0; j < kNumIntChannel; j++) {
        cell_bottom_right[j] += offset;
        *(feat_val++) = *(cell_bottom_right[j]) - *(cell_top_right[j]);
      }
    } else {
      for (int32_t j = 0; j < kNumIntChannel; j++) {
        cell_bottom_right[j] += offset;
        cell_top_left[j] = cell_top_right[j] - cell_width;
        cell_bottom_left[j] = cell_bottom_right[j] - cell_width;
        *(feat_val++) = *(cell_bottom_right[j]) + *(cell_top_left[j]) -
                        *(cell_top_right[j]) - *(cell_bottom_left[j]);
      }
    }

    // cells in following columns
    for (int32_t j = 1; j < feat.num_cell_per_row; j++) {
      for (int32_t k = 0; k < kNumIntChannel; k++) {
        cell_top_left[k] = cell_top_right[k];
        cell_top_right[k] += cell_width;

        cell_bottom_left[k] = cell_bottom_right[k];
        cell_bottom_right[k] += cell_width;

        *(feat_val++) = *(cell_bottom_right[k]) + *(cell_top_left[k]) -
                        *(cell_bottom_left[k]) - *(cell_top_right[k]);
      }
    }

    for (int32_t j = 0; j < kNumIntChannel; j++)
      cell_top_right[j] += offset;
  }
}

void SURFFeatureMap::NormalizeFeatureVectorL2(const int32_t* feat_vec,
    float* feat_vec_normed, int32_t len) const {
  double prod = 0.0;
  float norm_l2 = 0.0f;

  for (int32_t i = 0; i < len; i++)
    prod += static_cast<double>(feat_vec[i] * feat_vec[i]);
  if (prod != 0) {
    norm_l2 = static_cast<float>(std::sqrt(prod));
    for (int32_t i = 0; i < len; i++)
      feat_vec_normed[i] = feat_vec[i] / norm_l2;
  } else {
    for (int32_t i = 0; i < len; i++)
      feat_vec_normed[i] = 0.0f;
  }
}

}  // namespace fd
}  // namespace seeta
