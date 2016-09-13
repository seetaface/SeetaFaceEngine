/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Alignment module, containing codes implementing the
 * facial landmarks location method described in the following paper:
 *
 *
 *   Coarse-to-Fine Auto-Encoder Networks (CFAN) for Real-Time Face Alignment, 
 *   Jie Zhang, Shiguang Shan, Meina Kan, Xilin Chen. In Proceeding of the
 *   European Conference on Computer Vision (ECCV), 2014
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are firstly developed by Mengyi Liu (a Ph.D supervised by Prof. Shiguang Shan) and
 * further improved by Jie Zhang (a Ph.D supervised by Prof. Shiguang Shan) for more efficiency.
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

#pragma once

#include "stdio.h"
#include <string>
#include <cmath>

typedef unsigned char BYTE;

class SIFT{
 public:
  SIFT();
  ~SIFT();

  /** Initialize the SIFT extractor.
	  *  @param im_width The width of the input image
	  *  @param im_height The height of the input image
	  *  @param patch_size The size of one patch for extracting SIFT
	  *  @param grid_spacing The stride for extracting SIFT
	  */
  void InitSIFT(int im_width, int im_height, int patch_size, int grid_spacing);

  /** Compute SIFT feature
  *  @param gray_im A grayscale image
  *  @param[out] sift_feature The output SIFT feature
  */
  void CalcSIFT(BYTE* gray_im, double* sift_feature);

 private:
  /** Implement convolutional function "filter2" same in Matlab.
  *  @param gray_im A grayscale image
  *  @param kernel A convolutional kernel
  *  @param kernel_size The size of convolutional kernel
  *  @param[out] filter_im The output image map after convolution
  */
  void filter2(double* gray_im, double* kernel, int kernel_size, double* filter_im);

  /** Sparse convolution for speed-up
  *  @param gray_im A grayscale image
  *  @param kernel A convolutional kernel
  *  @param kernel_size The size of convolutional kernel
  *  @param[out] filter_im The output image map after sparse convolution
  */
  void SparseFilter2(double* gray_im, double* kernel, int kernel_size, double* filter_im);

  /** Calculate image orientation
  *  @param gray_im A grayscale image
  *  @param[out] image_orientation The output image orientation
  */
  void ImageOrientation(double* gray_im, double* image_orientation);

  /** Calculate image orientation
  *  @param image_orientation A image orientation map
  *  @param[out] conv_im The output convolutional image
  */
  void ConvImage(double* image_orientation, double* conv_im);

  private:
  struct SIFTParam
  {
	  int image_width;
	  int image_height;
	  int patch_size;
	  int grid_spacing;
	  int angle_nums;
	  int bin_nums;

	  int image_pixel;
	  int sample_nums;
	  int sample_pixel;
	  int patch_cnt_width;
	  int patch_cnt_height;
	  int patch_dims;
	  int image_dims;

	  int filter_size;
	  double sigma;
	  double alpha;
		
  };

  SIFTParam param;

  static double delta_gauss_x[25];
  static double delta_gauss_y[25];

};

