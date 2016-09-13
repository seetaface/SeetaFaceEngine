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

#include "classifier/surf_mlp.h"

#include <string>

namespace seeta {
namespace fd {

bool SURFMLP::Classify(float* score, float* outputs) {
  float* dest = input_buf_.data();
  for (size_t i = 0; i < feat_id_.size(); i++) {
    feat_map_->GetFeatureVector(feat_id_[i] - 1, dest);
    dest += feat_map_->GetFeatureVectorDim(feat_id_[i]);
  }
  output_buf_.resize(model_->GetOutputDim());
  model_->Compute(input_buf_.data(), output_buf_.data());

  if (score != nullptr)
    *score = output_buf_[0];
  if (outputs != nullptr) {
    std::memcpy(outputs, output_buf_.data(),
      model_->GetOutputDim() * sizeof(float));
  }

  return (output_buf_[0] > thresh_);
}

void SURFMLP::AddFeatureByID(int32_t feat_id) {
  feat_id_.push_back(feat_id);
}

void SURFMLP::AddLayer(int32_t input_dim, int32_t output_dim,
    const float* weights, const float* bias, bool is_output) {
  if (model_->GetLayerNum() == 0)
    input_buf_.resize(input_dim);
  model_->AddLayer(input_dim, output_dim, weights, bias, is_output);
}

}  // namespace fd
}  // namespace seeta
