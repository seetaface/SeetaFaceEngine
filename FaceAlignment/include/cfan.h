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
 * The codes are mainly developed by Jie Zhang (a Ph.D supervised by Prof. Shiguang Shan)
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
#include <cmath>
#include "sift.h"
#include "common.h"

class CCFAN{
 public:
  /** A constructor.
   *  Initialize basic parameters.
   */
  CCFAN(void);

  /** A destructor which should never be called explicitly.
   *  Release all dynamically allocated resources.
   */
  ~CCFAN(void);

  /** Initialize the facial landmark detection model.
    *  @param model_path Path of the model file, either absolute or relative to
    *                   the working directory.
    */
  void InitModel(const char *model_path);

  /** Detect five facial landmarks, i.e., two eye centers, nose tip and two mouth corners.
    *  @param gray_im A grayscale image
    *  @param im_width The width of the inpute image
    *  @param im_height The height of the inpute image
    *  @param face_loc The face bounding box
    *  @param[out] facial_loc The locations of detected facial points
    */
  void FacialPointLocate(const unsigned char *gray_im, int im_width, int im_height, seeta::FaceInfo face_loc, float *facial_loc);

 private:
  /** Extract shape indexed SIFT features.
    *  @param gray_im A grayscale image
    *  @param im_width The width of the inpute image
    *  @param im_height The height of the inpute image
    *  @param face_shape The locations of facial points
    *  @param patch_size The size of the patch used for extracting SIFT feature
    *  @param[out] sift_fea the extracted shape indexed SIFT features which are concatenated into a vector
    */
  void TtSift(const unsigned char *gray_im, int im_width, int im_height, float *face_shape, int patch_size, double *sift_fea);

  /** Extract a image patch which is centered at point(point_x, point_y) with a given patch size.
  *  @param gray_im A grayscale image
  *  @param im_width The width of the inpute image
  *  @param im_height The height of the inpute image
  *  @param point_x The X coordinate of one point
  *  @param point_y The Y coordinate of one point
  *  @param patch_size The size of the extracted patch
  *  @param[out] sub_img A grayscale image patch
  */
  void GetSubImg(const unsigned char *gray_im, int im_width, int im_height, float point_x, float point_y, int patch_size, BYTE *sub_img);

  /** Resize the image by bilinear interpolation.
    *  @param src_im A source image in grayscale
    *  @param src_width The width of the source image
    *  @param src_height The height of the source image
    *  @param[out] dst_im The target image in grayscale
    *  @param dst_width The width of the target image
    *  @param dst_height The height of the target image
    */
  bool ResizeImage(const unsigned char *src_im, int src_width, int src_height,
    unsigned char* dst_im, int dst_width, int dst_height);

 private:
  /*The number of facial points*/
  int pts_num_;
  /*The dimension of the shape indexed features*/
  int fea_dim_;
  /*The mean face shape containing five landmarks*/
  float *mean_shape_;

  /*The parameters of the first local stacked autoencoder network*/
  float **lan1_w_;
  float **lan1_b_;
  int *lan1_structure_;
  int lan1_size_;

  /*The parameters of the second local stacked autoencoder network*/
  float **lan2_w_;
  float **lan2_b_;
  int *lan2_structure_;
  int lan2_size_;

};

