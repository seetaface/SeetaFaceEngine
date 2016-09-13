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

#include "tform_maker_net.h"
#include "spatial_transform_net.h"
#include "aligner.h"

#include <vector>
namespace seeta {
Aligner::Aligner():
    crop_height_(256),
    crop_width_(256) {
  // build a aligner networks
  net_.reset(new CommonNet());
  HyperParam* common_param = net_->hyper_param();
  common_param->InsertInt("num_subnet", 2);
  common_param->InsertInt("num_in", 2);
  common_param->InsertInt("num_out", 1);
  net_->SetUp();
 
  std::shared_ptr<Net> tform_maker_net = net_->nets(0);
  tform_maker_net->SetFather(net_.get()); 
  // left_eye, right_eye, nose, left_mouse_corner and right_mouse_corner
  float std_points[10] = {
    89.3095, 72.9025,
    169.3095, 72.9025,
    127.8949, 127.0441,
    96.8796, 184.8907,
    159.1065, 184.7601,
  };
  HyperParam* tform_param = tform_maker_net->hyper_param();
  tform_param->InsertInt("points_num", 5);
  tform_maker_net->SetUp();
  Blob* tform_blob = tform_maker_net->params(0);
  tform_blob->CopyData(1, 5, 2, 1, std_points);

  std::shared_ptr<Net> align_net = net_->nets(1);
  align_net->SetFather(net_.get()); 

  HyperParam* align_param = align_net->hyper_param();
  align_param->InsertInt("new_height", crop_height_);
  align_param->InsertInt("new_width", crop_width_);
  align_param->InsertString("type", "bicubic");
  // Input with unpermuted mat data
  align_param->InsertInt("is_mat_data", 1);
  align_net->SetUp();
  
  net_->input_plugs(0).push_back(tform_maker_net->input_blobs(0));
  net_->input_plugs(1).push_back(align_net->input_blobs(0));
  tform_maker_net->output_plugs(0).push_back(align_net->input_blobs(1));
  align_net->output_plugs(0).push_back(net_->output_blobs(0));
}

Aligner::Aligner(int crop_height, int crop_width, std::string type):
    crop_height_(crop_height),
    crop_width_(crop_width) {
  // build a aligner networks
  net_.reset(new CommonNet());
  HyperParam* common_param = net_->hyper_param();
  common_param->InsertInt("num_subnet", 2);
  common_param->InsertInt("num_in", 2);
  common_param->InsertInt("num_out", 1);
  net_->SetUp();

  std::vector<std::shared_ptr<Net> >& sub_nets = net_->nets();
 
  sub_nets[0].reset(new TransformationMakerNet());
  sub_nets[1].reset(new SpatialTransformNet());
  std::shared_ptr<Net> tform_maker_net = net_->nets(0);
  tform_maker_net->SetFather(net_.get()); 
  // left_eye, right_eye, nose, left_mouse_corner and right_mouse_corner
  float std_points[10] = {
    89.3095, 72.9025,
    169.3095, 72.9025,
    127.8949, 127.0441,
    96.8796, 184.8907,
    159.1065, 184.7601,
  };
  for (int i = 0; i < 5; ++ i) {
    std_points[i * 2] *= crop_height_ / 256.0;
    std_points[i * 2 + 1] *= crop_width_ / 256.0;
  }
  HyperParam* tform_param = tform_maker_net->hyper_param();
  tform_param->InsertInt("points_num", 5);
  tform_maker_net->SetUp();
  Blob* tform_blob = tform_maker_net->params(0);
  tform_blob->CopyData(1, 5, 2, 1, std_points);

  std::shared_ptr<Net> align_net = net_->nets(1);
  align_net->SetFather(net_.get()); 

  HyperParam* align_param = align_net->hyper_param();
  align_param->InsertInt("new_height", crop_height_);
  align_param->InsertInt("new_width", crop_width_);
  align_param->InsertString("type", type);
  // Input with unpermuted mat data
  align_param->InsertInt("is_mat_data", 1);
  align_net->SetUp();
 
  //set connections 
  net_->input_plugs(0).push_back(tform_maker_net->input_blobs(0));
  net_->input_plugs(1).push_back(align_net->input_blobs(0));
  tform_maker_net->output_plugs(0).push_back(align_net->input_blobs(1));
  align_net->output_plugs(0).push_back(net_->output_blobs(0));
}

Aligner::~Aligner() {}

void Aligner::Alignment(const ImageData &src_img,
    const float* const points,
    Blob* const dst_blob) {
  Blob* const input_data = net_->input_blobs(1);
  input_data->reshape(1, src_img.num_channels, src_img.height, src_img.width);
  input_data->SetData();
  // input with mat::data avoid coping data
  memcpy(input_data->data().get(), src_img.data, input_data->count() * sizeof(unsigned char));
  /*input_data->CopyData(1, src_img.height, src_img.width, src_img.channels,
    src_img.data);
  input_data->Permute(1, 4, 2, 3);*/
  Blob* const input_point = net_->input_blobs(0);
  
  input_point->CopyData(1, 5, 2, 1, points);

  net_->Execute();
  dst_blob->SetData(*(net_->output_blobs(0)));
}

void Aligner::Alignment(const ImageData &src_img,
	  const float* const points,
	  const ImageData &dst_img) {
	Blob out_blob;
	Alignment(src_img, points, &out_blob);
	out_blob.Permute(1, 3, 4, 2);
	out_blob.CopyTo(dst_img.data);
}
}
