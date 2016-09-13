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

#include "eltwise_net.h"

void EltwiseNet::SetUp() {
  op_ = *(std::string*)(this->hyper_param()->param("eltwise_op"));
  if (op_ == "SCALE") {
    scale_ = *(float*)(this->hyper_param()->param("scale"));
    this->nets().resize(0);
    this->params().resize(0);
    this->input_blobs().resize(1);
    this->output_blobs().resize(1);
    this->input_plugs().resize(1);
    this->output_plugs().resize(1);
  }
  else if (op_ == "BAIS_ADDER") {
    this->nets().resize(0);
    this->params().resize(1);
    this->input_blobs().resize(1);
    this->output_blobs().resize(1);
    this->input_plugs().resize(1);
    this->output_plugs().resize(1);
  }
  else if (op_ == "CLOSE") {
    lower_ = *(float*)(this->hyper_param()->param("lower"));
    upper_ = *(float*)(this->hyper_param()->param("upper"));
    this->nets().resize(0);
    this->params().resize(0);
    this->input_blobs().resize(1);
    this->output_blobs().resize(1);
    this->input_plugs().resize(1);
    this->output_plugs().resize(1);
  }
  // SUM
  // PROD
  // MAX
}

void EltwiseNet::Execute() {
  CheckInput();
  if (op_ == "BAIS_ADDER") {
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

    int bn = (bias->num() != 1);
    int bc = (bias->channels() != 1);
    int bh = (bias->height() != 1);
    int bw = (bias->width() != 1);
    for (int n = 0, offset = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++ c) {
        for (int h = 0; h < height; ++ h) {
          for (int w = 0; w < width; ++ w, ++ offset) {
            dst_head[offset] = (*input)[offset] 
              + (*bias)[bias->offset(n*bn, c*bc, h*bh, w*bw)];
          }
        }
      }
    }

    output->CopyData(num, channels, height, width, dst_head);    
	delete[] dst_head;
  }
  else if (op_ == "SCALE") {
    const Blob* const input = this->input_blobs(0);
	LOG(DEBUG) << "input blob: (" << input->num() << ","
		<< input->channels() << ","
		<< input->height() << ","
		<< input->width() << ")";
    int count = input->count();
    Blob* const output = this->output_blobs(0);
    float* const dst_head = new float[count];
    for (int i = 0; i < count; ++ i)
      dst_head[i] = (*input)[i] * scale_;
    output->CopyData(input->num(), input->channels(), input->height(), 
                    input->width(), dst_head);
	delete[] dst_head;
  }
  else if (op_ == "CLOSE") {
    const Blob* const input = this->input_blobs(0);
	LOG(DEBUG) << "input blob: (" << input->num() << ","
		<< input->channels() << ","
		<< input->height() << ","
		<< input->width() << ")";
    int count = input->count();
    Blob* const output = this->output_blobs(0);
    float* const dst_head = new float[count];
    for (int i = 0; i < count; ++ i) {
      dst_head[i] = (*input)[i];
      dst_head[i] = std::min(dst_head[i], upper_);
      dst_head[i] = std::max(dst_head[i], lower_);
    }
    output->CopyData(input->num(), input->channels(), input->height(), 
                    input->width(), dst_head);
	delete[] dst_head;
  }
  // SUM
  // PROD
  // MAX
  CheckOutput();
}

REGISTER_NET_CLASS(Eltwise);
