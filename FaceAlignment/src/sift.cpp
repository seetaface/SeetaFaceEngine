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

#include "sift.h"
#include <string.h>

#define pi 3.1415926
double SIFT::delta_gauss_x[25] = 
{0.0284161904936934,0.0260724940559495,0,-0.0260724940559495,-0.0284161904936934,
0.127352530356230,0.116848811647003,0,-0.116848811647003,-0.127352530356230,
0.209968825675801,0.192651121218447,0,-0.192651121218447,-0.209968825675801,
0.127352530356230,0.116848811647003,0,-0.116848811647003,-0.127352530356230,
0.0284161904936934,0.0260724940559495,0,-0.0260724940559495,-0.0284161904936934};

double SIFT::delta_gauss_y[25] = 
{0.0284161904936934,0.127352530356230,0.209968825675801,0.127352530356230,0.0284161904936934,
0.0260724940559495,0.116848811647003,0.192651121218447,0.116848811647003,0.0260724940559495,
0,0,0,0,0,
-0.0260724940559495,-0.116848811647003,-0.192651121218447,-0.116848811647003,-0.0260724940559495,
-0.0284161904936934,-0.127352530356230,-0.209968825675801,-0.127352530356230,-0.0284161904936934};

SIFT::SIFT(void)
{
}


SIFT::~SIFT(void)
{
}

/** Initialize the SIFT extractor.
 *  @param im_width The width of the input image
 *  @param im_height The height of the input image
 *  @param patch_size The size of one patch for extracting SIFT
 *  @param grid_spacing The stride for extracting SIFT
 */
void SIFT::InitSIFT(int im_width, int im_height, int patch_size, int grid_spacing)
{
  param.image_width = im_width;
  param.image_height = im_height;
  param.patch_size = patch_size;
  param.grid_spacing = grid_spacing;
  param.angle_nums   = 8;
  param.bin_nums     = 4;

  param.image_pixel  = param.image_width * param.image_height;      
  param.sample_nums  = param.bin_nums * param.bin_nums;    
  param.sample_pixel = param.patch_size / param.bin_nums;   
  param.patch_cnt_width = (param.image_width - param.patch_size) / param.grid_spacing + 1;   
  param.patch_cnt_height = (param.image_height - param.patch_size) / param.grid_spacing + 1;  
  param.patch_dims = param.sample_nums * param.angle_nums;                      
  param.image_dims = param.patch_cnt_width * param.patch_cnt_height * param.patch_dims;  

  param.filter_size = 5;
  param.sigma = 1;
  param.alpha = 3;	
}

/** Implement convolutional function "filter2" same in Matlab.
 *  @param gray_im A grayscale image
 *  @param kernel A convolutional kernel
 *  @param kernel_size The size of convolutional kernel
 *  @param[out] filter_im The output image map after convolution
 */
void SIFT::filter2(double* gray_im, double* kernel, int kernel_size, double* filter_im)
{
  // Padding the image
  int pad_size = (kernel_size - 1) / 2;
  double* gray_img_ex = new double[(param.image_width + (kernel_size - 1)) * (param.image_height + (kernel_size - 1))];
	
  for(int i = 0; i < pad_size; i++)
  {
	  for (int j = 0; j < param.image_width + (kernel_size - 1); j++)
	  {
		  gray_img_ex[i * (param.image_width + (kernel_size - 1)) + j] = 0;
	  }
  }

  for (int i = param.image_height + pad_size; i < param.image_height + (kernel_size - 1); i++)
  {
	  for (int j = 0; j < param.image_width + (kernel_size - 1); j++)
	  {
		  gray_img_ex[i * (param.image_width + (kernel_size - 1)) + j] = 0;
	  }
  }

  for(int i = pad_size; i < param.image_height + pad_size; i++)
  {
	  for(int j = 0; j < pad_size; j++)
	  {
		  gray_img_ex[i * (param.image_width + (kernel_size - 1)) + j] = 0;
	  }
	  for (int j = param.image_width + pad_size; j < param.image_width + (kernel_size - 1); j++)
	  {
		  gray_img_ex[i * (param.image_width + (kernel_size - 1)) + j] = 0;
	  }
	  for(int j = pad_size; j < param.image_width + pad_size; j++)
	  {
		  gray_img_ex[i * (param.image_width + (kernel_size - 1)) + j] = gray_im[(i - pad_size) * param.image_width + (j - pad_size)];
	  }
  }

  // Sliding filter on padding image
  for(int i = 0; i < param.image_height; i++)
  {
	  for(int j = 0; j < param.image_width; j++)
	  {
		  double tmp = 0.000000;

		  for (int ki = 0; ki < kernel_size; ki++)
		  {
			  for (int kj = 0; kj < kernel_size; kj++)
			  {
				  double tmp1 = gray_img_ex[(i + ki) * (param.image_width + (kernel_size - 1)) + (j + kj)];
				  tmp += tmp1 * kernel[ki * kernel_size + kj];
			  }
		  }
		  filter_im[i * param.image_width + j] = tmp;
	  }
  }
  delete [] gray_img_ex;
}

/** Sparse convolution for speed-up
 *  @param gray_im A grayscale image
 *  @param kernel A convolutional kernel
 *  @param kernel_size The size of convolutional kernel
 *  @param[out] filter_im The output image map after sparse convolution
 */
void SIFT::SparseFilter2(double* gray_im, double* kernel, int kernel_size, double* filter_im)
{
  // Padding the image
  int pad_size = (kernel_size-1)/2;
  double* gray_img_ex = new double[(param.image_width + (kernel_size-1)) * (param.image_height + (kernel_size-1))];
	
  for(int i = 0; i < pad_size; i++)
  {
	  for(int j = 0; j < param.image_width + (kernel_size-1); j++)
	  {
		  gray_img_ex[i * (param.image_width + (kernel_size-1)) + j] = 0;
	  }
  }

  for(int i = param.image_height + pad_size; i < param.image_height + (kernel_size-1); i++)
  {
	  for(int j = 0; j < param.image_width + (kernel_size-1); j++)
	  {
		  gray_img_ex[i * (param.image_width + (kernel_size-1)) + j] = 0;
	  }
  }

  for(int i = pad_size; i < param.image_height + pad_size; i++)
  {
	  for(int j = 0; j < pad_size; j++)
	  {
		  gray_img_ex[i * (param.image_width + (kernel_size-1)) + j] = 0;
	  }
	  for(int j = param.image_width + pad_size; j < param.image_width + (kernel_size-1); j++) 
	  {
		  gray_img_ex[i * (param.image_width + (kernel_size-1)) + j] = 0;
	  }
	  for(int j = pad_size; j < param.image_width + pad_size; j++)
	  {
		  gray_img_ex[i * (param.image_width + (kernel_size - 1)) + j] = gray_im[(i - pad_size) * param.image_width + (j - pad_size)];
	  }
  }

  // Sliding filter on padding image
  for(int i = 0; i < param.image_height; i += param.sample_pixel)
  {
	  for(int j = 0; j < param.image_width; j += param.sample_pixel)
	  {
		  double tmp = 0.000000;

		  for(int ki = 0; ki < kernel_size; ki++)
		  {
			  for(int kj = 0; kj < kernel_size; kj++)
			  {
				  double tmp1 = gray_img_ex[(i + ki) * (param.image_width + (kernel_size - 1)) + (j + kj)];
				  tmp += tmp1 * kernel[ki * kernel_size + kj];
			  }
		  }
		  filter_im[i * param.image_width + j] = tmp;
	  }
  }
  delete [] gray_img_ex;
}

/** Calculate image orientation
 *  @param image_orientation A image orientation map
 *  @param[out] conv_im The output convolutional image
 */
void SIFT::ConvImage(double* image_orientation, double* conv_im)
{
  double* weight = new double[param.patch_size];
  double* kernel = new double[param.patch_size * param.patch_size];

  for(int k = 0; k < param.patch_size; k++)
  {
	  weight[k] = abs(k - double(param.patch_size - 1)/2)/(param.sample_pixel);

	  if(weight[k] <= 1)
		  weight[k] = 1 - weight[k];
	  else
		  weight[k] = 0;
  }

  for(int i = 0; i < param.patch_size; i++)
  {
	  for(int j = 0; j < param.patch_size; j++)
	  {
		  kernel[i * param.patch_size + j] = weight[i] * weight[j];
	  }
  }

  double* angle_im = new double[param.image_pixel];
  double* angle_conv_im = new double[param.image_pixel];

  for(int index = 0; index < param.angle_nums; index++)
  {
	  memset(angle_im, 0, param.image_pixel * sizeof(double));
	  memcpy(angle_im, &image_orientation[index * param.image_pixel], param.image_pixel * sizeof(double));
	  SparseFilter2(angle_im, kernel, param.patch_size, angle_conv_im);
	  memcpy(&conv_im[index * param.image_pixel], angle_conv_im, param.image_pixel * sizeof(double));
  }
	
  delete [] weight;
  delete [] kernel;
  delete [] angle_im;
  delete [] angle_conv_im;
}

/** Compute SIFT feature
 *  @param gray_im A grayscale image
 *  @param[out] sift_feature The output SIFT feature
 */
void SIFT::CalcSIFT(BYTE* gray_im, double* sift_feature)
{
  double* lf_gray_im = new double[param.image_pixel];
  double max = 0.000001;
  for (int pt = 0; pt < param.image_pixel; pt++)
  {
	  lf_gray_im[pt] = gray_im[pt];
	  if (lf_gray_im[pt] > max)
		  max = lf_gray_im[pt];
  }

  for (int pt = 0; pt < param.image_pixel; pt++)
  {
	  lf_gray_im[pt] = lf_gray_im[pt] / max;
  }

  double* im_orientation = new double[param.image_pixel * param.angle_nums];
  double* conv_im = new double[param.image_pixel * param.angle_nums];
  memset(conv_im, 0, param.image_pixel * param.angle_nums * sizeof(double));

  ImageOrientation(lf_gray_im, im_orientation);
  ConvImage(im_orientation, conv_im);

  // Generate denseSIFT feature vector
  double* patch_feature = new double[param.patch_dims];
  int patch_cnt = 0;

  // Sliding windows on overlapping patches. (px,py) are centroids
  for (int location_x = param.patch_size / 2; location_x <= param.image_height - (param.patch_size / 2); location_x += param.grid_spacing)
  {
	  for (int location_y = param.patch_size / 2; location_y <= param.image_width - (param.patch_size / 2); location_y += param.grid_spacing)
	  {
		  memset(patch_feature, 0, param.patch_dims * sizeof(double));

		  double l2_norm = 0.000001;
		  int Point_cnt = 0;

		  for (int p_x = -param.patch_size / 2; p_x <= param.patch_size / 2 - param.sample_pixel; p_x += param.sample_pixel)
		  {
			  for (int p_y = -param.patch_size / 2; p_y <= param.patch_size / 2 - param.sample_pixel; p_y += param.sample_pixel)
			  {
				  int i = location_x + p_x;
				  int j = location_y + p_y;

				  for (int index = 0; index < param.angle_nums; index++)
				  {
					  patch_feature[Point_cnt] = conv_im[index * param.image_pixel + j * param.image_height + i];
					  l2_norm += pow(patch_feature[Point_cnt], 2);
					  Point_cnt += 1;
				  }
			  }
		  }
		  // Patch-wise L2-norm
		  double norm = 1.0 / sqrt(l2_norm);
		  for (int pt = 0; pt < param.patch_dims; pt++)
		  {
			  patch_feature[pt] = patch_feature[pt] * norm;
		  }

		  memcpy(&sift_feature[patch_cnt * param.patch_dims], patch_feature, param.patch_dims * sizeof(double));
		  patch_cnt += 1;
	  }
  }

  delete[] lf_gray_im;
  delete[] im_orientation;
  delete[] conv_im;
  delete[] patch_feature;
}


/** Calculate image orientation
 *  @param image_orientation A image orientation map
 *  @param[out] conv_im The output convolutional image
 */
void SIFT::ImageOrientation(double* gray_im, double* image_orientation)
{
  double* im_vert_edge = new double[param.image_pixel];
  double* im_hori_edge = new double[param.image_pixel];

  filter2(gray_im, delta_gauss_x, param.filter_size, im_vert_edge);
  filter2(gray_im, delta_gauss_y, param.filter_size, im_hori_edge);

  double* im_magnitude = new double[param.image_pixel];
  double* im_cos_theta = new double[param.image_pixel];
  double* im_sin_theta = new double[param.image_pixel];

  for (int i = 0; i < param.image_height; i++)
  {
	  for (int j = 0; j < param.image_width; j++)
	  {
		  double tmpV = im_vert_edge[i * param.image_width + j];
		  double tmpH = im_hori_edge[i * param.image_width + j];
		  double tmpMagnitude = sqrt(pow(tmpV, 2) + pow(tmpH, 2));
		  im_magnitude[i * param.image_width + j] = tmpMagnitude;
		  im_cos_theta[i * param.image_width + j] = tmpV / tmpMagnitude;
		  im_sin_theta[i * param.image_width + j] = tmpH / tmpMagnitude;
	  }
  }

  delete[] im_vert_edge;
  delete[] im_hori_edge;

  double cos_array[8];
  double sin_array[8];
  cos_array[0] = 1.0;
  cos_array[1] = 0.7071;
  cos_array[2] = 0.0;
  cos_array[3] = -0.7071;
  cos_array[4] = -1.0;
  cos_array[5] = -0.7071;
  cos_array[6] = 0.0;
  cos_array[7] = 0.7071;

  sin_array[0] = 0.0;
  sin_array[1] = 0.7071;
  sin_array[2] = 1.0;
  sin_array[3] = 0.7071;
  sin_array[4] = 0.0;
  sin_array[5] = -0.7071;
  sin_array[6] = -1.0;
  sin_array[7] = -0.7071;
  for (int index = 0; index < param.angle_nums; index++)
  {
	  for (int pt = 0; pt < param.image_pixel; pt++)
	  {
		  double tmp1 = im_cos_theta[pt] * cos_array[index] + im_sin_theta[pt] * sin_array[index];
		  double tmp = pow(tmp1,3);

		  if (tmp > 0)
			  tmp = tmp;
		  else
			  tmp = 0;
		  image_orientation[index * param.image_pixel + pt] = tmp * im_magnitude[pt];
	  }
  }

  delete[] im_magnitude;
  delete[] im_cos_theta;
  delete[] im_sin_theta;
}
