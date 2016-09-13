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

#include "blob.h"

#include <cstdio>
#include <cstring>
#include <fstream>

Blob::Blob() {
  data_ = nullptr;
  shape_.clear();
}

Blob::Blob(const Blob &source) {
  if (data_)
    data_ = nullptr;
  shape_ = source.shape();
  count_ = source.count();
  data_ = source.data();
}

Blob::Blob(int n, int c, int h, int w) {
  if (data_)
    data_ = nullptr;
  shape_.resize(4);
  shape_[0] = n;
  shape_[1] = c;
  shape_[2] = h;
  shape_[3] = w;
  count_ = n * c * h * w;
}

Blob::Blob(int n, int c, int h, int w, float* data) {
  if (data_)
    data_ = nullptr;
  shape_.resize(4);
  shape_[0] = n;
  shape_[1] = c;
  shape_[2] = h;
  shape_[3] = w;
  count_ = n * c * h * w;
  data_.reset(new float[count_], std::default_delete<float[]>());
  memcpy(data_.get(), data, count_ * sizeof(float));
}

Blob::Blob(FILE* file) {
  shape_.resize(4);
  CHECK_EQ(fread(&(shape_[0]), sizeof(int), 1, file), 1);
  CHECK_EQ(fread(&(shape_[1]), sizeof(int), 1, file), 1);
  CHECK_EQ(fread(&(shape_[2]), sizeof(int), 1, file), 1);
  CHECK_EQ(fread(&(shape_[3]), sizeof(int), 1, file), 1);
  count_ = shape_[0] * shape_[1] * shape_[2] * shape_[3];
  data_.reset(new float[count_], std::default_delete<float[]>());
  CHECK_EQ(fread(data_.get(), sizeof(float), count_, file), count_);
}

Blob::~Blob() {
  shape_.clear();
  count_ = 0;
  data_ = nullptr;
}

void Blob::reshape(int n, int c, int h, int w) {
  shape_.resize(4);
  shape_[0] = n;
  shape_[1] = c;
  shape_[2] = h;
  shape_[3] = w;
  count_ = n * c * h * w;
  data_ = nullptr; 
}

void Blob::Permute(int dim1, int dim2, int dim3, int dim4) {
  // todo: check permute
  std::vector<int> dim(4), redim(4), idx(4);
  dim[0] = dim1 - 1; redim[dim[0]] = 0;
  dim[1] = dim2 - 1; redim[dim[1]] = 1;
  dim[2] = dim3 - 1; redim[dim[2]] = 2;
  dim[3] = dim4 - 1; redim[dim[3]] = 3;
  float* tmp = new float[count_];
  float* dat = data_.get();
  int cnt = 0;
  for (idx[0] = 0; idx[0] < shape_[dim[0]]; ++ idx[0]) {
    for (idx[1] = 0; idx[1] < shape_[dim[1]]; ++ idx[1]) {
      for (idx[2] = 0; idx[2] < shape_[dim[2]]; ++ idx[2]) {
        for (idx[3] = 0; idx[3] < shape_[dim[3]]; ++ idx[3]) {
          tmp[cnt] = dat[offset(idx[redim[0]], idx[redim[1]], idx[redim[2]],
              idx[redim[3]])];
          cnt ++ ;
        }
      }
    }
  }
  std::vector<int> tmp_shape(4);
  for (int i = 0; i < 4; ++ i)
    tmp_shape[i] = shape_[dim[i]];
  for (int i = 0; i < 4; ++ i)
    shape_[i] = tmp_shape[i];
  memcpy(dat, tmp, sizeof(float) * count_);
  delete tmp;
}

void Blob::Release() {
  data_ = nullptr;
}

void Blob::SetData() {
  if (!data_)
    data_.reset(new float[count_], std::default_delete<float[]>());
}


void Blob::SetData(Blob &source) {
  if (data_)
    data_ = nullptr;
  shape_ = source.shape();
  count_ = num() * channels() * height() * width();
  data_ = source.data();
}

void Blob::SetData(int n, int c, int h, int w) {
  if (data_)
    data_ = nullptr;
  shape_.resize(4);
  shape_[0] = n;
  shape_[1] = c;
  shape_[2] = h;
  shape_[3] = w;
  count_ = n * c * h * w;
  data_.reset(new float[count_], std::default_delete<float[]>());
}


void Blob::CopyData(int n, int c, int h, int w, const float* const data) {
  if (data_)
    data_ = nullptr;
  shape_.resize(4);
  shape_[0] = n;
  shape_[1] = c;
  shape_[2] = h;
  shape_[3] = w;
  count_ = n * c * h * w;
  data_.reset(new float[count_], std::default_delete<float[]>());
  memcpy(data_.get(), data, count_ * sizeof(float));
}

void Blob::CopyData(int n, int c, int h, int w, 
    const unsigned char* const data) {
  if (data_)
    data_ = nullptr;
  shape_.resize(4);
  shape_[0] = n;
  shape_[1] = c;
  shape_[2] = h;
  shape_[3] = w;
  count_ = n * c * h * w;
  data_.reset(new float[count_], std::default_delete<float[]>());
  float* data_head = data_.get();
  for (int i = 0; i < count_; ++ i)
    data_head[i] = data[i];
}

void Blob::CopyTo(unsigned char* const data) {
  const float* const data_head = data_.get();
  for (int i = 0; i < count_; ++ i)
    data[i] = std::min(255.0f, std::max(0.0f, data_head[i]));
}

void Blob::CopyTo(float* const data) {
	float* data_head = data_.get();
	memcpy(data, data_head, count_ * sizeof(float));
}

void Blob::ToFile(const std::string file_name) {
  std::ofstream ofs;
  ofs.open(file_name, std::ofstream::out);
  float* data = data_.get();
  for (int i = 0; i < count_; ++ i) {
    ofs << data[i] << " ";
  }
  ofs << std::endl;
  ofs.close();
}

void Blob::ToBinaryFile(const std::string file_name) {
  FILE* file = nullptr;
  fopen_s(&file, file_name.c_str(), "wb");
  if (file == nullptr) {
    LOG(ERROR) << file_name << " not exist!";
    exit(0);
  }
  for (int i = 0; i < 4; ++ i) {
    if (i < shape_.size())
      fwrite(&shape_[i], 1, sizeof(int), file);
    else {
      int one = 1;
      fwrite(&one, 1, sizeof(int), file);
    }
  }
  fwrite(data_.get(), count_, sizeof(float), file);
  fclose(file);
}


