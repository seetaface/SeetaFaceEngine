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

#include "pad_net.h"

#include <algorithm>

void PadNet::SetUp() {
  left_ = right_ = bottom_ = top_ = *(int*)(this->hyper_param()->param("pad"));
  // check input and output blob size
  this->nets().resize(0);
  this->params().resize(0);
  this->input_blobs().resize(1);
  this->output_blobs().resize(1);
  this->input_plugs().resize(1);
  this->output_plugs().resize(1);
}

void PadNet::Execute() {
  CheckInput();
  const Blob* const input = this->input_blobs(0);
  Blob* const output = this->output_blobs(0);
  int src_w = input->width();
  int src_h = input->height();
  int num = input->num();
  int channels = input->channels();

  LOG(DEBUG) << "input blob: (" << num << "," << channels << "," << src_h 
    << "," << src_w << ")";

  int dst_w = src_w + left_ + right_;
  int dst_h = src_h + top_ + bottom_;

  int height = std::min(src_h, dst_h);
  int width = std::min(src_w, dst_w);

  output->reshape(num, channels, dst_h, dst_w);

  float* const data = new float[num * channels * dst_h * dst_w]();

  for (int n = 0; n < num; ++ n) {
    int src_off = input->offset(n), dst_off = output->offset(n);
    for (int c = 0; c < channels; ++ c) {
      if (top_ < 0 && left_ < 0) {
        src_off = input->offset(n, c, -top_, -left_);
      }
      else if(top_ >= 0 && left_ >= 0) {
        dst_off = output->offset(n, c, top_, left_);
      }
      else {
        LOG(ERROR) << "Illegal padding parameters!";
        exit(0);
      }
      for (int h = 0; h < height; ++ h) {
        memcpy(data + dst_off, input->data().get() + src_off, 
               sizeof(float) * width);
        src_off += src_w;
        dst_off += dst_w;
      }
    }
  }
  output->CopyData(num, channels, dst_h, dst_w, data);
  delete[] data;
  CheckOutput();
}

REGISTER_NET_CLASS(Pad);
