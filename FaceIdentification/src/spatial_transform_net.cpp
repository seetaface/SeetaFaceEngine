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

#include "spatial_transform_net.h"
#include "math.h"

#include <algorithm>

void SpatialTransformNet::SetUp() {
  type_ = *(std::string *)(this->hyper_param()->param("type"));
  new_height_ = *(int *)(this->hyper_param()->param("new_height"));
  new_width_ = *(int *)(this->hyper_param()->param("new_width"));
  if (this->hyper_param()->has_param("is_mat_data")) {
    is_mat_data_ = *(int *)(this->hyper_param()->param("is_mat_data"));
  }
  else {
    is_mat_data_ = false;
  }
  // check input and output blob size
  this->input_blobs().resize(2);
  this->output_blobs().resize(1);
  this->input_plugs().resize(2);
  this->output_plugs().resize(1);
}

void SpatialTransformNet::Execute() {
  CheckInput();
  const Blob* const input = this->input_blobs(0);
  const Blob* const theta = this->input_blobs(1);
  Blob* const output = this->output_blobs(0);

  CHECK_EQ(input->num(), theta->num());

  int num = input->num();
  int channels = input->channels();
  
  int src_w = input->width();
  int src_h = input->height();

  LOG(DEBUG) << "Input blobs: (" << num << "," << channels << "," << src_h
    << "," << src_w << ")";

  int dst_h = new_height_;
  int dst_w = new_width_;

  float* input_data = input->data().get();
  const float* theta_data = theta->data().get();
  int tform_size = theta->count() / num;

  LOG(DEBUG) << "Theta: [" << theta_data[0] << "," << theta_data[1] << ","
    << theta_data[2] << "," << theta_data[3] << "," << theta_data[4] << ","
    << theta_data[5] << "]";

  CHECK_EQ(tform_size, 6);

  output->SetData(num, channels, dst_h, dst_w);

  float* output_data = output->data().get();

  for (int n = 0; n < num; ++ n) {
    double scale = sqrt(theta_data[0] * theta_data[0] 
        + theta_data[3] * theta_data[3]);
    for (int x = 0; x < dst_h; ++ x)
      for (int y = 0; y < dst_w; ++ y) {
        // Get the source position of each point on the destination feature map.
        double src_y = theta_data[0] * y + theta_data[1] * x + theta_data[2];
        double src_x = theta_data[3] * y + theta_data[4] * x + theta_data[5];
        for (int c = 0; c < channels; ++ c) {
          if (!is_mat_data_) {
            output_data[output->offset(n, c, x, y)]
              = Sampling(input_data + input->offset(n, c), src_h, src_w,
              src_x, src_y, 1.0 / scale);
          }
          else {
            output_data[output->offset(n, c, x, y)]
              = Sampling(reinterpret_cast<unsigned char*>(
                  input_data + input->offset(n)), c, src_h, src_w,
              this->output_blobs(0)->channels(), src_x, src_y, 1.0 / scale);
          }
        }
      }
    theta_data += tform_size;
  }

  LOG(DEBUG) << "Output blobs: (" << this->output_blobs(0)->num() << "," 
    << this->output_blobs(0)->channels() << "," 
    << this->output_blobs(0)->height() << ","
    << this->output_blobs(0)->width() << ")";
  CheckOutput();
}

double SpatialTransformNet::Sampling(const float* const feat_map, int H, int W, 
    double x, double y, double scale) {
  if (type_ == "linear") {
    // bilinear subsampling
    int ux = floor(x), uy = floor(y);
    double ans = 0;
    if (ux >= 0 && ux < H - 1 && uy >= 0 && uy < W - 1) {
      int offset = ux * W + uy;
      double cof_x = x - ux;
      double cof_y = y - uy;
      ans = (1 - cof_y) * feat_map[offset] + cof_y * feat_map[offset + 1];
      ans = (1 - cof_x) * ans + cof_x * ((1 - cof_y) * feat_map[offset + W] 
          + cof_y * feat_map[offset + W + 1]);
    }
    return ans;
  }
  else if (type_ == "bicubic") { // Need to be sped up
    // bicubic subsampling
    double ans = 0;
    if (x >= 0 && x < H && y >= 0 && y < W) {
      scale = std::min(scale, double(1.0));
      double kernel_width =  std::max(8.0, 4.0 / scale); // bicubic kernel width
      std::vector<double> weights_x, weights_y; 
      std::vector<int>  indices_x, indices_y;
      weights_x.reserve(5), indices_x.reserve(5);
      weights_y.reserve(5), indices_y.reserve(5);
      // get indices and weight along x axis
      for (int ux = ceil(x - kernel_width / 2); 
          ux <= floor(x + kernel_width / 2); ++ ux) {
        indices_x.push_back(std::max(std::min(H - 1, ux), 0));
        weights_x.push_back(Cubic((x - ux) * scale));
      }
      // get indices and weight along y axis
       for (int uy = ceil(y - kernel_width / 2); 
          uy <= floor(y + kernel_width / 2); ++ uy) {
        indices_y.push_back(std::max(std::min(W - 1, uy), 0));
        weights_y.push_back(Cubic((y - uy) * scale));
      }     
      // normalize the weights
      Norm(weights_x);
      Norm(weights_y);
      double val = 0;
	  int lx = weights_x.size(), ly = weights_y.size();
      for (int i = 0; i < lx; ++ i) {
        if (i == 0 || indices_x[i] != indices_x[i - 1]) {
          val = 0;
          int offset = indices_x[i] * W;
          for (int j = 0; j < ly; ++ j) {
            val += feat_map[offset + indices_y[j]] * weights_y[j];
          }
        }
        ans += val * weights_x[i];
      }
    }
    return ans;
  }
}

double SpatialTransformNet::Sampling(const unsigned char* const feat_map, 
    int c, int H, int W, int C, double x, double y, double scale) {
  if (type_ == "linear") {
    // bilinear subsampling
    int ux = floor(x), uy = floor(y);
    double ans = 0;
    if (ux >= 0 && ux < H - 1 && uy >= 0 && uy < W - 1) {
      int offset = (ux * W + uy) * C + c;
      double cof_x = x - ux;
      double cof_y = y - uy;
      ans = (1 - cof_y) * feat_map[offset] + cof_y * feat_map[offset + C];
      ans = (1 - cof_x) * ans + cof_x * ((1 - cof_y) * feat_map[offset + W * C]
        + cof_y * feat_map[offset + W * C + C]);
    }
    return ans;
  }
  else if (type_ == "bicubic") { // Need to be sped up
    // bicubic subsampling
    double ans = 0;
    if (x >= 0 && x < H && y >= 0 && y < W) {
      scale = std::min(scale, double(1.0));
      double kernel_width = std::max(8.0, 4.0 / scale); // bicubic kernel width
      std::vector<double> weights_x, weights_y;
      std::vector<int>  indices_x, indices_y;
      weights_x.reserve(5), indices_x.reserve(5);
      weights_y.reserve(5), indices_y.reserve(5);
      // get indices and weight along x axis
      for (int ux = ceil(x - kernel_width / 2);
        ux <= floor(x + kernel_width / 2); ++ux) {
        indices_x.push_back(std::max(std::min(H - 1, ux), 0));
        weights_x.push_back(Cubic((x - ux) * scale));
      }
      // get indices and weight along y axis
      for (int uy = ceil(y - kernel_width / 2);
        uy <= floor(y + kernel_width / 2); ++uy) {
        indices_y.push_back(std::max(std::min(W - 1, uy), 0));
        weights_y.push_back(Cubic((y - uy) * scale));
      }
      // normalize the weights
      Norm(weights_x);
      Norm(weights_y);
      double val = 0;
      int lx = weights_x.size(), ly = weights_y.size();
      for (int i = 0; i < lx; ++i) {
        if (i == 0 || indices_x[i] != indices_x[i - 1]) {
          val = 0;
          int offset = indices_x[i] * W * C;
          for (int j = 0; j < ly; ++j) {
            val += feat_map[offset + indices_y[j] * C + c] * weights_y[j];
          }
        }
        ans += val * weights_x[i];
      }
    }
    return ans;
  }
}

double SpatialTransformNet::Cubic(double x) {
  double ax = fabs(x), ax2, ax3;
  ax2 = ax * ax;
  ax3 = ax2 * ax;
  if (ax <= 1) return 1.5 * ax3 - 2.5 * ax2 + 1;
  if (ax <= 2) return -0.5 * ax3 + 2.5 * ax2 -4 * ax + 2;
  return 0;
}

REGISTER_NET_CLASS(SpatialTransform);
