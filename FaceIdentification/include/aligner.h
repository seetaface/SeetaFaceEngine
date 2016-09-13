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
 * The codes are mainly developed by Wanglong Wu(a Ph.D supervised by Prof. Shiguang Shan)
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
#ifndef ALIGNER_H_
#define ALIGNER_H_

#include "common.h"
#include "common_net.h"
#include "net.h"

namespace seeta {
class Aligner {
 public:
  Aligner();
  Aligner(int crop_height, int crop_width, std::string type = "bicubic");
  ~Aligner();
  // Alignment and return to a ImageData
  void Alignment(const ImageData &src_img, 
      const float* const llpoint, 
      const ImageData &dst_img); 
  // Alignment and return to a Blob
  void Alignment(const ImageData &src_img, 
      const float* const llpoint, 
      Blob* const dst_blob); 

  void set_height(int height) { crop_height_ = height; }
  void set_width(int width) {crop_width_ = width; }

  int crop_height() { return crop_height_; }
  int crop_width() { return crop_width_; }
 private:
  int crop_height_;
  int crop_width_;
  std::shared_ptr<Net> net_;
};
} // namespace 
#endif // ALIGNER_H_
