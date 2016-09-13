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

#ifndef SEETA_FD_DETECTOR_H_
#define SEETA_FD_DETECTOR_H_

#include <cstdint>
#include <string>
#include <vector>

#include "common.h"
#include "util/image_pyramid.h"

namespace seeta {
namespace fd {

class Detector {
 public:
  Detector() {}
  virtual ~Detector() {}

  virtual bool LoadModel(const std::string & model_path) = 0;
  virtual std::vector<seeta::FaceInfo> Detect(seeta::fd::ImagePyramid* img_pyramid) = 0;

  virtual void SetWindowSize(int32_t size) {}
  virtual void SetSlideWindowStep(int32_t step_x, int32_t step_y) {}

  DISABLE_COPY_AND_ASSIGN(Detector);
};

}  // namespace fd
}  // namespace seeta

#endif  // SEETA_FD_DETECTOR_H_
