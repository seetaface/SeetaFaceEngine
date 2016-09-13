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

#ifndef BLOB_H_
#define BLOB_H_
#include "log.h"
#include "viplnet.h"

#include <vector>
#include <string>
#include <memory>
#include <algorithm>

// basic computational unit

class Blob {
 public:
  Blob();
  Blob(const Blob &source);
  Blob(int n, int c, int h, int w);
  Blob(int n, int c, int h, int w, float* data);
  Blob(FILE* file);

  virtual ~Blob();
  void reshape(int n, int c, int h, int w);
  void Permute(int dim1, int dim2, int dim3, int dim4);
  void Release();
  inline int offset(const int n, const int c=0, const int h=0,
      const int w=0) const {
    if (n < 0 || n >= num() || c < 0 || c >= channels() ||
        h < 0 || h >= height() || w < 0 || w >= width()) {
      LOG(ERROR)
        << "Index: (" << n << ", " << c << ", " << h << ", " << w << "), "
        << "Bound: [" << num() << ", " << channels() << ", " << height()
        <<", " << width() << "].";
      exit(0);
    }
    return ((n * channels() + c) * height() + h) * width() + w;
  }
  void SetData();
  void SetData(Blob &source);
  void SetData(int n, int c, int h, int w);
  void CopyData(int n, int c, int h, int w, const float* const data);
  // copy data from unsigned char
  void CopyData(int n, int c, int h, int w, const unsigned char* const  data);
  // copy data to unsigned char
  void CopyTo(unsigned char* const data);
  // copy data to float
  void CopyTo(float* const data);
  void ToFile(const std::string file_name);
  void ToBinaryFile(const std::string file_name);

  inline const float operator [](int i) const {
    return data_.get()[i];
  }
  inline float & operator [](int i) {
    return data_.get()[i];
  }

  inline int num() const { return shape(0); }
  inline int channels() const { return shape(1); }
  inline int height() const { return shape(2); }
  inline int width() const { return shape(3); }
  inline std::vector<int> shape() const { return shape_; }
  int count() const { return count_; }
  std::shared_ptr<float> data() const { return data_; }
 private:
  inline int shape(int index) const {
    return index < shape_.size() ? shape_[index] : 1;
  }
  std::shared_ptr<float> data_;
  std::vector<int> shape_;
  int count_;
};

#endif // BLOB_H_
