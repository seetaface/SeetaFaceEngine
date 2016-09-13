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

#include "face_identification.h"
#include "recognizer.h"

#include "math_functions.h"

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

namespace seeta {
FaceIdentification::FaceIdentification(const char* model_path) {
  recognizer = new Recognizer(model_path);
}

FaceIdentification::~FaceIdentification() {
  delete recognizer;
}

uint32_t FaceIdentification::LoadModel(const char* model_path) {
  return recognizer->LoadModel(model_path);
}

uint32_t FaceIdentification::feature_size() {
  return recognizer->feature_size();
}

uint32_t FaceIdentification::crop_width() {
  return recognizer->crop_width();
}

uint32_t FaceIdentification::crop_height() {
  return recognizer->crop_height();
}

uint32_t FaceIdentification::crop_channels() {
  return recognizer->crop_channels();
}

uint8_t FaceIdentification::CropFace(const ImageData &src_image,
    const FacialLandmark* llpoint,
    const ImageData &dst_image) {
  if (src_image.num_channels != recognizer->crop_channels() ||
    src_image.data == NULL) {
    std::cout << "Face Recognizer: Error input image." << std::endl;
    return 0;
  }
  if (dst_image.data == NULL) {
    std::cout << "Face Recognizer: Error output image." << std::endl;
    return 0;
  }
  float point_data[10];
  for (int i = 0; i < 5; ++i) {
	  point_data[i * 2] = llpoint[i].x;
	  point_data[i * 2 + 1] = llpoint[i].y;
  }
  recognizer->Crop(src_image, point_data, dst_image);
  return 1;
}

uint8_t FaceIdentification::ExtractFeature(const ImageData &cropImg, 
    FaceFeatures const feats) {
  if (feats == NULL) {
    std::cout << "Face Recognizer: 'feats' must be initialized with size \
           of GetFeatureSize(). " << std::endl;
    return 0;
  }
  recognizer->ExtractFeature(cropImg.data, feats);
  return 1;
}

uint8_t FaceIdentification::ExtractFeatureWithCrop(const ImageData &src_image, 
    const FacialLandmark* llpoint, 
    FaceFeatures const feats) {
  float point_data[10];
  for (int i = 0; i < 5; ++i) {
	point_data[i * 2] = llpoint[i].x;
	point_data[i * 2 + 1] = llpoint[i].y;
  }
  recognizer->ExtractFeatureWithCrop(src_image, point_data, feats);
  return 1;
}

float FaceIdentification::CalcSimilarity(FaceFeatures const fc1,
    FaceFeatures const fc2,
    long dim) {
  if (dim == -1) {
    dim = recognizer->feature_size();
  }
  return simd_dot(fc1, fc2, dim)
	  / (sqrt(simd_dot(fc1, fc1, dim))
	  * sqrt(simd_dot(fc2, fc2, dim)));
}
}
