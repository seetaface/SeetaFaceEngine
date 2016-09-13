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

#ifndef NET_H_
#define NET_H_

#include "viplnet.h"
#include "blob.h"
#include "log.h"
#include "hyper_param.h"

#include <vector>

class Net {
 public:
  Net();
  virtual ~Net();
  // initialize the networks from a binary file
  virtual void SetUp();
  // execute the networks
  virtual void Execute() = 0;

  // check input blobs
  virtual void CheckInput();

  // check output blobs
  virtual void CheckOutput();

  virtual void Release() {
    for (int i = 0; i < output_blobs_.size(); ++i)
      output_blobs_[i].Release();
  }

  Net* const father() {
    return father_;
  }
  void SetFather(Net* father) {
    father_ = father;
  }
  std::vector<std::shared_ptr<Net> >& nets() {
    return nets_;
  }
  std::shared_ptr<Net> nets(int i) {
    return nets_[i];
  }
  std::vector<Blob>& input_blobs() {
    return input_blobs_;
  }
  Blob* input_blobs(int i) {
    return &(input_blobs_[i]);
  }
  std::vector<Blob>& output_blobs() {
    return output_blobs_;
  }
  Blob* output_blobs(int i) {
    return &(output_blobs_[i]);
  }
  std::vector<std::vector<Blob*> >& output_plugs() {
    return output_plugs_;
  }
  std::vector<Blob*>& output_plugs(int i) {
    return output_plugs_[i];
  }
  std::vector<std::vector<Blob*> >& input_plugs() {
    return input_plugs_;
  }
  std::vector<Blob*>& input_plugs(int i) {
    return input_plugs_[i];
  }
  HyperParam* hyper_param() {
    return &hyper_params_;
  }
  std::vector<Blob>& params() {
    return params_;
  }
  Blob* params(int i) {
    return &(params_[i]);
  }
  // count the number of unreleased output blobs
  inline int num_output() {
    int count = 0;
    for (int i = 0; i < output_blobs_.size(); ++ i)
      if (output_plugs_[i].size() == 0) count += 1;
    return count;
  }
 protected:
  // father net
  Net* father_;
  // the limit of net name size
  static const int MAX_NET_NAME_SIZE = 50;
  // net name
  std::string name_;

  // input and output blobs
  std::vector<Blob> input_blobs_;
  std::vector<Blob> output_blobs_;

  // subnet of the networks
  std::vector<std::shared_ptr<Net> > nets_;

  // plugs
  std::vector<std::vector<Blob*> > output_plugs_;
  std::vector<std::vector<Blob*> > input_plugs_;

  // params in the networks
  HyperParam hyper_params_;
  std::vector<Blob> params_;
};

#endif //NET_H_
