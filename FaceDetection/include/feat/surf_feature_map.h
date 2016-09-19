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

#ifndef SEETA_FD_FEAT_SURF_FEATURE_MAP_H_
#define SEETA_FD_FEAT_SURF_FEATURE_MAP_H_

#include <cstdint>
#include <cstring>
#include <vector>

#include "common.h"
#include "feature_map.h"
#include "util/math_func.h"

namespace seeta {
namespace fd {

typedef struct SURFFeature {
  seeta::Rect patch;
  int32_t num_cell_per_row;
  int32_t num_cell_per_col;
} SURFFeature;

class SURFFeaturePool {
 public:
  SURFFeaturePool()
      : sample_width_(40), sample_height_(40),
        patch_move_step_x_(16), patch_move_step_y_(16), patch_size_inc_step_(1),
        patch_min_width_(16), patch_min_height_(16) {}

  ~SURFFeaturePool() {}

  void Create();
  void AddPatchFormat(int32_t width, int32_t height, int32_t num_cell_per_row,
      int32_t num_cell_per_col);

  inline bool empty() const { return pool_.empty(); }
  inline std::size_t size() const { return pool_.size(); }

  inline std::vector<SURFFeature>::const_iterator begin() const {
    return pool_.begin();
  }

  inline std::vector<SURFFeature>::const_iterator end() const {
    return pool_.end();
  }

  inline const SURFFeature & operator[](std::size_t idx) const {
    return pool_[idx];
  }

 private:
  void AddAllFeaturesToPool(int32_t width, int32_t height,
      int32_t num_cell_per_row, int32_t num_cell_per_col);

  typedef struct SURFPatchFormat {
    /**< aspect ratio, s.t. GCD(width, height) = 1 */
    int32_t width;
    int32_t height;

    /**< cell partition */
    int32_t num_cell_per_row;
    int32_t num_cell_per_col;
  } SURFPatchFormat;

  int32_t sample_width_;
  int32_t sample_height_;
  int32_t patch_move_step_x_;
  int32_t patch_move_step_y_;
  int32_t patch_size_inc_step_; /**< incremental step of patch width and */
                                /**< height when build feature pool      */
  int32_t patch_min_width_;
  int32_t patch_min_height_;

  std::vector<SURFFeature> pool_;
  std::vector<SURFPatchFormat> format_;
};

class SURFFeatureMap : public FeatureMap {
 public:
  SURFFeatureMap() : buf_valid_reset_(false) { InitFeaturePool(); }
  virtual ~SURFFeatureMap() {}

  virtual void Compute(const uint8_t* input, int32_t width, int32_t height);

  inline virtual void SetROI(const seeta::Rect & roi) {
    roi_ = roi;
    if (buf_valid_reset_) {
      std::memset(buf_valid_.data(), 0, buf_valid_.size() * sizeof(int32_t));
      buf_valid_reset_ = false;
    }
  }

  inline int32_t GetFeatureVectorDim(int32_t feat_id) const {
    return (feat_pool_[feat_id].num_cell_per_col *
      feat_pool_[feat_id].num_cell_per_row * kNumIntChannel);
  }

  void GetFeatureVector(int32_t featID, float* featVec);

 private:
  void InitFeaturePool();
  void Reshape(int32_t width, int32_t height);

  void ComputeGradientImages(const uint8_t* input);
  void ComputeGradX(const int32_t* input);
  void ComputeGradY(const int32_t* input);
  void ComputeIntegralImages();
  void Integral();
  void MaskIntegralChannel();

  inline void FillIntegralChannel(const int32_t* src, int32_t ch) {
    int32_t* dest = int_img_.data() + ch;
    int32_t len = width_ * height_;
    for (int32_t i = 0; i < len; i++) {
      *dest = *src;
      *(dest + 2) = *src;
      dest += kNumIntChannel;
      src++;
    }
  }

  void ComputeFeatureVector(const SURFFeature & feat, int32_t* feat_vec);
  void NormalizeFeatureVectorL2(const int32_t* feat_vec, float* feat_vec_normed,
    int32_t len) const;

  /**
   * Number of channels should be divisible by 4.
   */
  void VectorCumAdd(int32_t* x, int32_t len, int32_t num_channel);

  static const int32_t kNumIntChannel = 8;

  bool buf_valid_reset_;

  std::vector<int32_t> grad_x_;
  std::vector<int32_t> grad_y_;
  std::vector<int32_t> int_img_;
  std::vector<int32_t> img_buf_;
  std::vector<std::vector<int32_t> > feat_vec_buf_;
  std::vector<std::vector<float> > feat_vec_normed_buf_;
  std::vector<int32_t> buf_valid_;

  seeta::fd::SURFFeaturePool feat_pool_;
};

}  // namespace fd
}  // namespace seeta

#endif  // SEETA_FD_FEAT_SURF_FEATURE_MAP_H_
