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

#include "inner_product_net.h"
#include "math_functions.h"

void InnerProductNet::SetUp() {
  // check input and output blob size
  this->input_blobs().resize(1);
  this->output_blobs().resize(1);
  this->input_plugs().resize(1);
  this->output_plugs().resize(1);
  this->params().resize(1);
}

void InnerProductNet::Execute() {
  CheckInput();
  const Blob* const input = this->input_blobs(0); // src_num * vec_len
  const Blob* const weight = this->params(0);  // dst_channels * vec_len
  Blob* const output = this->output_blobs(0); // src_num * dst_channels
  
  int src_num = input->num();
  int src_channels = input->channels();
  int src_h = input->height();
  int src_w = input->width();
  int dst_channels = weight->num();
  
  LOG(DEBUG) << "input blob: (" <<src_num << "," << src_channels << "," << src_h 
    << "," << src_w << ")";

  const int vec_len = src_channels * src_h * src_w;
  float* const dst_head = new float[src_num * dst_channels];
  const float* src_data = input->data().get();
  for (int sn = 0, didx = 0; sn < src_num; ++sn) {
    const float* weight_data = weight->data().get();
    for (int dc = 0; dc < dst_channels; ++dc) {
      dst_head[didx++] = simd_dot(src_data, weight_data, vec_len);
      weight_data += vec_len;
    } // for dc
    
    src_data += vec_len;
  } // for sn
  
  output->CopyData(src_num, dst_channels, 1, 1, dst_head);
  delete[] dst_head;
  LOG(DEBUG) << "output blob: (" << output->num() << "," << output->channels() 
    << "," << output->height() << "," << output->width() << ")";
  CheckOutput();
}

REGISTER_NET_CLASS(InnerProduct);
