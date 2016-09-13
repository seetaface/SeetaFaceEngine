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
 * The codes are mainly developed by Chunrui Han(a Ph.D supervised by Prof. Shiguang Shan)
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

#include "tform_maker_net.h"

void TransformationMakerNet::SetUp() {
  points_num_ = *(int*)(this->hyper_param()->param("points_num"));
  // check input and output blob size
  this->input_blobs().resize(1);
  this->output_blobs().resize(1);
  this->input_plugs().resize(1);
  this->output_plugs().resize(1);
  this->params().resize(1);
}

void TransformationMakerNet::Execute() {
  const float EPS = 1e-4;
  const int TFORM_SIZE = 6;
  CheckInput();
  CHECK_EQ(points_num_, this->input_blobs(0)->channels());

  Blob* const input = this->input_blobs(0);
  Blob* const param = this->params(0);
  const float* feat_points = input->data().get();
  const float* std_points = param->data().get();
  float* out_data = new float[input->num() * TFORM_SIZE];

  for (int n = 0; n < input->num(); ++ n) {
    double sum_x = 0, sum_y = 0;
    double sum_u = 0, sum_v = 0;
    double sum_xx_yy = 0;
    double sum_ux_vy = 0;
    double sum_vx__uy = 0;
    for (int c = 0; c < points_num_; ++ c) {
      int x_off = n * points_num_ * 2 + c * 2;
      int y_off = x_off + 1; 
      sum_x += std_points[c * 2];
      sum_y += std_points[c * 2 + 1];
      sum_u += feat_points[x_off];
      sum_v += feat_points[y_off];
      sum_xx_yy += std_points[c * 2] * std_points[c * 2] + 
                   std_points[c * 2 + 1] * std_points[c * 2 + 1];
      sum_ux_vy += std_points[c * 2] * feat_points[x_off] +
                   std_points[c * 2 + 1] * feat_points[y_off];
      sum_vx__uy += feat_points[y_off] * std_points[c * 2] -
                    feat_points[x_off] * std_points[c * 2 + 1];
    }
    CHECK_GT(sum_xx_yy, EPS);
    double q = sum_u - sum_x * sum_ux_vy / sum_xx_yy 
                     + sum_y * sum_vx__uy / sum_xx_yy;
    double p = sum_v - sum_y * sum_ux_vy / sum_xx_yy 
                     - sum_x * sum_vx__uy / sum_xx_yy;

    double r = points_num_ - (sum_x * sum_x + sum_y * sum_y) / sum_xx_yy;
    
    CHECK_TRUE(r > EPS || r < -EPS) << " r = " << r;

    double a = (sum_ux_vy - sum_x * q / r - sum_y * p / r) / sum_xx_yy;

    double b = (sum_vx__uy + sum_y * q / r - sum_x * p / r) / sum_xx_yy; 

    double c = q / r;

    double d = p / r;

    float* tform = out_data + n * TFORM_SIZE;
    tform[0] = tform[4] = a;
    tform[1] = -b;
    tform[3] = b;
    tform[2] = c;
    tform[5] = d;
  }
  this->output_blobs(0)->CopyData(input->num(), TFORM_SIZE, 1, 1, out_data);
  delete[] out_data;
  CheckOutput();
}


