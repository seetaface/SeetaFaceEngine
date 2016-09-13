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
 * The codes are mainly developed by Mengru Zhang(a M.S. supervised by Prof. Shiguang Shan)
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

#include "bias_adder_net.h"

void BiasAdderNet::SetUp() {
  this->input_blobs().resize(1);
  this->output_blobs().resize(1);
  this->input_plugs().resize(1);
  this->output_plugs().resize(1);
  this->params().resize(1);
}

void BiasAdderNet::Execute() {
  CheckInput();

  const Blob* const input = this->input_blobs(0);
  const Blob* const bias = this->params(0);
  Blob* const output = this->output_blobs(0);

  int channels = bias->channels();
  CHECK_EQ(channels, input->channels());

  int height = input->height();
  int width = input->width();
  int num = input->num();
  
  LOG(DEBUG) << "input blob: (" << num << "," << input->channels() << "," 
    << height << "," << width << ")";
  LOG(DEBUG) << "bias blob: (" << bias->num() << "," << bias->channels() 
    << "," << bias->height() << "," << bias->width() << ")";
  
  float* const dst_head = new float[num*channels*height*width];

  int size = height * width;
  for (int n = 0, offset = 0; n < num; ++n) {
    for (int ichannel = 0; ichannel < channels; ++ichannel) {
      for(int i = 0; i < size; ++i, ++offset) {
        dst_head[offset] = (*input)[offset] + (*bias)[ichannel];
      }
    }
  }

  output->CopyData(num, channels, height, width, dst_head);
  delete []dst_head;
  CheckOutput();
}

REGISTER_NET_CLASS(BiasAdder);
