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

#ifndef RECOGNIZER_H_
#define RECOGNIZER_H_

#include "face_identification.h"
#include "aligner.h"
#include "common_net.h"

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

namespace seeta {
class FaceIdentification::Recognizer {
public:
  Recognizer(const char* model_path = nullptr) {
    if (model_path == nullptr) {
      crop_width_ = 256;
      crop_height_ = 256;
      crop_channels_ = 3;
      net_ = nullptr;
      aligner_.reset(new Aligner(crop_height_, crop_width_, "linear"));
      feat_size_ = 0;
    } 
    else {
	  crop_width_ = 0;
	  crop_height_ = 0;
	  crop_channels_ = 0;
	  net_ = nullptr;
	  aligner_ = nullptr;
	  feat_size_ = 0;
      LoadModel(model_path);
    }
  }

  ~Recognizer() {
  }

  uint8_t LoadModel(const char* model_path) {
    FILE* file = nullptr;
    if (fopen_s(&file, model_path, "rb") != 0) {
      std::cout<< model_path << " not exist!" << std::endl;
      exit(-1);
    }
    CHECK_EQ(fread(&crop_channels_, sizeof(int), 1, file), 1); 
    CHECK_EQ(fread(&crop_height_, sizeof(int), 1, file), 1);
    CHECK_EQ(fread(&crop_width_, sizeof(int), 1, file), 1);
	  CHECK_EQ(fread(&feat_size_, sizeof(int), 1, file), 1);
    /*if (!aligner_ || crop_height_ != aligner_->CropHeight() 
		||  crop_width_ != aligner_->CropHeight())*/
    aligner_.reset(new Aligner(crop_height_, crop_width_, "linear"));
    net_ = CommonNet::Load(file);
    return 1;
  }

  uint8_t Crop(const ImageData &src_img, const float* const points,
      const ImageData &dst_img) {
    aligner_->Alignment(src_img, points, dst_img);
    return 1;
  }

  uint8_t ExtractFeature(unsigned char* const u_data, float* const feat, 
      int n = 1) {
    net_->input_blobs(0)->CopyData(n, crop_height_, crop_width_, 
		                           crop_channels_, u_data);
    net_->input_blobs(0)->Permute(1, 4, 2, 3);
    net_->Execute();

    net_->output_blobs(0)->CopyTo(feat);
    net_->Release();
    return 1;
  }

  uint8_t ExtractFeatureWithCrop(const ImageData &src_img, 
      float* const points, float* const feat) {
    // crop
    Blob crop_blob;
    aligner_->Alignment(src_img, points, &crop_blob);
    // extract feature
    net_->input_blobs(0)->SetData(crop_blob);
    net_->Execute();

    net_->output_blobs(0)->CopyTo(feat);
    net_->Release();
	return 1;
  }

  uint8_t ExtractFeature(unsigned char* const u_data, uint16_t* const feat) {
	//To do: compress features 
    return 1;
  }
  
  uint32_t crop_width() { return crop_width_; }
  uint32_t crop_height() { return crop_height_; }
  uint32_t crop_channels() { return crop_channels_; }
  uint32_t crop_mem_size() { return crop_width_ * crop_height_ * crop_channels_; }
  uint32_t feature_size() { return feat_size_; }

private:
  std::shared_ptr<Net> net_;
  std::shared_ptr<Aligner> aligner_;
  uint32_t crop_width_;
  uint32_t crop_height_;
  uint32_t crop_channels_;

  uint32_t feat_size_;
  uint8_t isLoadModel() {
    return net_ != nullptr;
  }
};
}
#endif // RECOGNIZER_H_
