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

#include "util/image_pyramid.h"

namespace seeta {
namespace fd {

const seeta::ImageData* ImagePyramid::GetNextScaleImage(float* scale_factor) {
  if (scale_factor_ >= min_scale_) {
    if (scale_factor != nullptr)
      *scale_factor = scale_factor_;

    width_scaled_ = static_cast<int32_t>(width1x_ * scale_factor_);
    height_scaled_ = static_cast<int32_t>(height1x_ * scale_factor_);

    seeta::ImageData src_img(width1x_, height1x_);
    seeta::ImageData dest_img(width_scaled_, height_scaled_);
    src_img.data = buf_img_;
    dest_img.data = buf_img_scaled_;
    seeta::fd::ResizeImage(src_img, &dest_img);
    scale_factor_ *= scale_step_;

    img_scaled_.data = buf_img_scaled_;
    img_scaled_.width = width_scaled_;
    img_scaled_.height = height_scaled_;
    return &img_scaled_;
  } else {
    return nullptr;
  }
}

void ImagePyramid::SetImage1x(const uint8_t* img_data, int32_t width,
    int32_t height) {
  if (width > buf_img_width_ || height > buf_img_height_) {
    delete[] buf_img_;

    buf_img_width_ = width;
    buf_img_height_ = height;
    buf_img_ = new uint8_t[width * height];
  }

  width1x_ = width;
  height1x_ = height;
  std::memcpy(buf_img_, img_data, width * height * sizeof(uint8_t));
  scale_factor_ = max_scale_;
  UpdateBufScaled();
}

void ImagePyramid::UpdateBufScaled() {
  if (width1x_ == 0 || height1x_ == 0)
    return;

  int32_t max_width = static_cast<int32_t>(width1x_ * max_scale_ + 0.5);
  int32_t max_height = static_cast<int32_t>(height1x_ * max_scale_ + 0.5);

  if (max_width > buf_scaled_width_ || max_height > buf_scaled_height_) {
    delete[] buf_img_scaled_;

    buf_scaled_width_ = max_width;
    buf_scaled_height_ = max_height;
    buf_img_scaled_ = new uint8_t[max_width * max_height];

    img_scaled_.data = nullptr;
    img_scaled_.width = 0;
    img_scaled_.height = 0;
  }
}

}  // namespace fd
}  // namespace seeta
