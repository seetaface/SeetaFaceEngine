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

#ifndef SPATIAL_TRANSFORM_NET_H_
#define SPATIAL_TRANSFORM_NET_H_

#include "net.h"
#include "net_factory.h"

class SpatialTransformNet : public Net {
 public:
  SpatialTransformNet() : Net() {}
  virtual ~SpatialTransformNet() {}
  virtual void SetUp();
  virtual void Execute();
 
 protected:
  // sampling for common blob data
  virtual double Sampling(const float* const feat_map, int H, int W, double x, 
      double y, double scale = 1.0);

  // sampling for cv::Mat::data
  virtual double Sampling(const unsigned char* const feat_map, int c, int H, 
      int W, int C, double x, double y, double scale = 1.0);

  virtual double Cubic(double x);

  inline void Norm(std::vector<double>& weights) {
    float sum = 0;
    for (int i = 0; i < weights.size(); ++ i)
      sum += weights[i];
    for (int i = 0; i < weights.size(); ++ i)
      weights[i] /= sum;
  }

  // transformation type: linear or bicubic
  std::string type_;
  // whether input with cv::Mat::data
  int is_mat_data_;
  // output feature map height and width
  int new_height_, new_width_;
};

#endif //SPATIAL_TRANSFORM_NET_H_
