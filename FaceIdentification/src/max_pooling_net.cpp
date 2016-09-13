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

#include "max_pooling_net.h"
#include <cfloat>
#include <math.h>

#include <algorithm>

void MaxPoolingNet::SetUp() {
  kernel_h_ = kernel_w_ = 
      *(int*)(this->hyper_param()->param("kernel_size"));
  stride_h_ = stride_w_ = 
      *(int*)(this->hyper_param()->param("stride"));

  // check input and output blob size
  this->input_blobs().resize(1);
  this->output_blobs().resize(1);
  this->input_plugs().resize(1);
  this->output_plugs().resize(1);
}

void MaxPoolingNet::Execute() {
  // *** Argument *** //
  const float MIN_THRESHOLD = 0.0f;
  // *** //
  
  CheckInput();
  const Blob* const input = this->input_blobs(0);
  Blob* const output = this->output_blobs(0);
  
  int src_h = input->height();
  int src_w = input->width();
  int num = input->num();
  int channels = input->channels();

  int dst_h = static_cast<int>(ceil(static_cast<float>(
    src_h - kernel_h_) / stride_h_)) + 1;
  int dst_w = static_cast<int>(ceil(static_cast<float>(
    src_w - kernel_w_) / stride_w_)) + 1;

  int dst_count = num * channels * dst_h * dst_w;
  float* const dst_head = new float[dst_count];
  const float* src_data = input->data().get();
  float* dst_data = dst_head;
  int src_channel_off = src_h * src_w;
  int dst_channel_off = dst_h * dst_w;
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int dh = 0, hstart = 0; dh < dst_h;
            ++dh, hstart += stride_h_) {
        int hend = std::min(hstart + kernel_h_, src_h);
        for (int dw = 0, wstart = 0; dw < dst_w;
            ++dw, wstart += stride_w_) {
          int wend = std::min(wstart + kernel_w_, src_w);
          int didx = dh * dst_w + dw;

          float max_val = MIN_THRESHOLD;
          for (int sh = hstart; sh < hend; ++sh) {
            for (int sw = wstart; sw < wend; ++sw) {
              int sidx = sh * src_w + sw;
              if (src_data[sidx] > max_val) {
                max_val = src_data[sidx];
              }
            } // for sw
          } // for sh
          dst_data[didx] = max_val;

        } // for dw
      } // for dh

      src_data += src_channel_off;
      dst_data += dst_channel_off;
    } // for c
  } // for n

  output->CopyData(num, channels, dst_h, dst_w, dst_head);
  delete[] dst_head;
  CheckOutput();
}

REGISTER_NET_CLASS(MaxPooling);
