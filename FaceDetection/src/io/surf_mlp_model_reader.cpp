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

#include "io/surf_mlp_model_reader.h"

#include <istream>

#include "classifier/surf_mlp.h"

namespace seeta {
namespace fd {

bool SURFMLPModelReader::Read(std::istream* input,
    seeta::fd::Classifier* model) {
  bool is_read = false;
  seeta::fd::SURFMLP* surf_mlp = dynamic_cast<seeta::fd::SURFMLP*>(model);
  int32_t num_layer;
  int32_t num_feat;
  int32_t input_dim;
  int32_t output_dim;
  float thresh;

  input->read(reinterpret_cast<char*>(&num_layer), sizeof(int32_t));
  if (num_layer <= 0) {
    is_read = false;  // @todo handle the errors and the following ones!!!
  }
  input->read(reinterpret_cast<char*>(&num_feat), sizeof(int32_t));
  if (num_feat <= 0) {
    is_read = false;
  }

  feat_id_buf_.resize(num_feat);
  input->read(reinterpret_cast<char*>(feat_id_buf_.data()),
    sizeof(int32_t) * num_feat);
  for (int32_t i = 0; i < num_feat; i++)
    surf_mlp->AddFeatureByID(feat_id_buf_[i]);

  input->read(reinterpret_cast<char*>(&thresh), sizeof(float));
  surf_mlp->SetThreshold(thresh);
  input->read(reinterpret_cast<char*>(&input_dim), sizeof(int32_t));
  if (input_dim <= 0) {
    is_read = false;
  }

  for (int32_t i = 1; i < num_layer; i++) {
    input->read(reinterpret_cast<char*>(&output_dim), sizeof(int32_t));
    if (output_dim <= 0) {
      is_read = false;
    }

    int32_t len = input_dim * output_dim;
    weights_buf_.resize(len);
    input->read(reinterpret_cast<char*>(weights_buf_.data()),
      sizeof(float) * len);

    bias_buf_.resize(output_dim);
    input->read(reinterpret_cast<char*>(bias_buf_.data()),
      sizeof(float) * output_dim);

    if (i < num_layer - 1) {
      surf_mlp->AddLayer(input_dim, output_dim, weights_buf_.data(),
        bias_buf_.data());
    } else {
      surf_mlp->AddLayer(input_dim, output_dim, weights_buf_.data(),
        bias_buf_.data(), true);
    }
    input_dim = output_dim;
  }

  is_read = !input->fail();

  return is_read;
}

}  // namespace fd
}  // namespace seeta
