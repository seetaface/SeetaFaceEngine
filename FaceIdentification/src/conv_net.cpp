/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Detection module, containing codes implementing the
 * face detection method described in the following paper:
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
 * The codes are mainly developed by Zining Xu(a M.S. supervised by Prof. Shiguang Shan)
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

#include "conv_net.h"
#include "math_functions.h"
#ifdef __VIPL_LOG__
#include <ctime>
#endif

void ConvNet::SetUp() {
  stride_h_ = stride_w_ =
      *(int*)(this->hyper_param()->param("stride"));

  // check input and output blob size
  this->input_blobs().resize(1);
  this->output_blobs().resize(1);
  this->input_plugs().resize(1);
  this->output_plugs().resize(1);
  this->params().resize(1);
}

void ConvNet::Execute() {
#ifdef __VIPL_LOG__
  double t_start, t_end, scan_time, math_time;
#endif
  // *** Argument *** //
  const bool is_binary = false;
  // *** //

  CheckInput();
  const Blob* const input = this->input_blobs(0);
  const Blob* const weight = this->params(0);
  Blob* const output = this->output_blobs(0);

  int src_num = input->num();
  int src_channels = input->channels();
  int src_h = input->height();
  int src_w = input->width();
  int dst_channels = weight->num();
  int kernel_h = weight->height();
  int kernel_w = weight->width();

  LOG(DEBUG) << "input blob: (" <<src_num << "," << src_channels << "," << src_h
    << "," << src_w << ")";

  int dst_h = (src_h - kernel_h) / stride_h_ + 1;
  int dst_w = (src_w - kernel_w) / stride_w_ + 1;
  int end_h = src_h - kernel_h + 1;
  int end_w = src_w - kernel_w + 1;
  int dst_size = dst_h * dst_w;
  int kernel_size = src_channels * kernel_h * kernel_w;

  const int src_num_offset = src_channels * src_h * src_w;
  float* const dst_head =
      new float[src_num * dst_size * dst_channels];
  float* const mat_head =
      new float[dst_size * kernel_size];

  const float* src_data = input->data().get();
  float* dst_data = dst_head;
  int didx = 0;
#ifdef __VIPL_LOG__
  scan_time = math_time = 0;
#endif
  for (int sn = 0; sn < src_num; ++sn) {
#ifdef __VIPL_LOG__
    t_start = clock();
#endif
    float* mat_data = mat_head;
    for (int sh = 0; sh < end_h; sh += stride_h_) {
      for (int sw = 0; sw < end_w; sw += stride_w_) {
        for (int sc = 0; sc < src_channels; ++sc) {
          int src_off = (sc * src_h + sh) * src_w + sw;
          for (int hidx = 0; hidx < kernel_h; ++hidx) {
            memcpy(mat_data, src_data + src_off,
                    sizeof(float) * kernel_w);
            mat_data += kernel_w;
            src_off += src_w;
          }
        } // for sc
      } // for sw
    } // for sh
    src_data += src_num_offset;
#ifdef __VIPL_LOG__
    t_end = clock();
    scan_time += t_end - t_start;

    t_start = clock();
#endif

    const float* weight_head = weight->data().get();
    matrix_procuct(mat_head, weight_head, dst_data, dst_size, dst_channels,
      kernel_size, true, false);
#ifdef __VIPL_LOG__
    t_end = clock();
    math_time += t_end - t_start;
#endif
    dst_data += dst_channels * dst_size;
  } // for sn

#ifdef __VIPL_LOG__
  LOG(INFO) << "scan time: " << scan_time / CLOCKS_PER_SEC * 1000 << "ms";
  LOG(INFO) << "math time: " << math_time / CLOCKS_PER_SEC * 1000 << "ms";
#endif
  output->CopyData(src_num, dst_channels, dst_h, dst_w, dst_head);
  delete[] mat_head;
  delete[] dst_head;

  LOG(DEBUG) << "output blob: (" << output->num() << "," << output->channels()
    << "," << output->height() << "," << output->width() << ")";
  CheckOutput();
}

REGISTER_NET_CLASS(Conv);
