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


#include "cfan.h"
#include <string.h>
#include <algorithm>
/** A constructor.
  *  Initialize basic parameters.
  */
CCFAN::CCFAN(void)
{
  pts_num_ = 5;
  fea_dim_ = pts_num_ * 128;

  lan1_w_ = NULL;
  lan1_b_ = NULL;
  lan1_structure_ = NULL;

  lan2_w_ = NULL;
  lan2_b_ = NULL;
  lan2_structure_ = NULL;

  mean_shape_ = NULL;
}

/** A destructor which should never be called explicitly.
  *  Release all dynamically allocated resources.
  */
CCFAN::~CCFAN(void)
{
  if (lan1_structure_ != NULL)
  {
    delete[]lan1_structure_;
    lan1_structure_ = NULL;
  }
  if (lan1_w_ != NULL)
  {
    for (int i = 0; i < lan1_size_ - 1; i++)
    {
      delete[](lan1_w_[i]);
      delete[](lan1_b_[i]);
    }
    delete[]lan1_w_;
    delete[]lan1_b_;
    lan1_w_ = NULL;
    lan1_b_ = NULL;
  }

  if (lan2_structure_ != NULL)
  {
    delete[]lan2_structure_;
  }
  if (lan2_w_ != NULL)
  {
    for (int i = 0; i < lan2_size_ - 1; i++)
    {
      delete[](lan2_w_[i]);
      delete[](lan2_b_[i]);
    }
    delete[]lan2_w_;
    delete[]lan2_b_;
    lan2_w_ = NULL;
    lan2_b_ = NULL;
  }

  if (mean_shape_)
  {
    delete[]mean_shape_;
    mean_shape_ = NULL;
  }
}

/** Initialize the facial landmark detection model.
  *  @param model_path Path of the model file, either absolute or relative to
  *                   the working directory.
  */
void CCFAN::InitModel(const char *model_path)
{
  /*Open the model file*/
  FILE *fp = fopen(model_path, "rb+");
  mean_shape_ = new float[pts_num_ * 2];
  fread(mean_shape_, sizeof(float), pts_num_ * 2, fp);

  /*Load the parameters of the first local stacked autoencoder network*/
  fread(&lan1_size_, sizeof(int), 1, fp);
  lan1_structure_ = new int[lan1_size_];
  fread(lan1_structure_, sizeof(int), lan1_size_, fp);

  lan1_w_ = new float *[lan1_size_ - 1];
  lan1_b_ = new float *[lan1_size_ - 1];
  for (int i = 0; i < lan1_size_ - 1; i++)
  {
    int layer_size = lan1_structure_[i] * lan1_structure_[i + 1];
    lan1_w_[i] = new float[layer_size];
    fread(lan1_w_[i], sizeof(float), layer_size, fp);

    lan1_b_[i] = new float[lan1_structure_[i + 1]];
    fread(lan1_b_[i], sizeof(float), lan1_structure_[i + 1], fp);
  }

  /*Load the parameters of the second local stacked autoencoder network*/
  fread(&lan2_size_, sizeof(int), 1, fp);
  lan2_structure_ = new int[lan2_size_];
  fread(lan2_structure_, sizeof(int), lan2_size_, fp);

  lan2_w_ = new float *[lan2_size_ - 1];
  lan2_b_ = new float *[lan2_size_ - 1];
  for (int i = 0; i < lan2_size_ - 1; i++)
  {
    int layer_size = lan2_structure_[i] * lan2_structure_[i + 1];
    lan2_w_[i] = new float[layer_size];
    fread(lan2_w_[i], sizeof(float), layer_size, fp);

    lan2_b_[i] = new float[lan2_structure_[i + 1]];
    fread(lan2_b_[i], sizeof(float), lan2_structure_[i + 1], fp);
  }
  fclose(fp);
}

/** Detect five facial landmarks, i.e., two eye centers, nose tip and two mouth corners.
  *  @param gray_im A grayscale image
  *  @param im_width The width of the inpute image
  *  @param im_height The height of the inpute image
  *  @param face_loc The face bounding box
  *  @param[out] facial_loc The locations of detected facial points
  */
void CCFAN::FacialPointLocate(const unsigned char *gray_im, int im_width, int im_height, seeta::FaceInfo face_loc, float *facial_loc)
{
  int sift_patch_size = 32;
  int left_x = face_loc.bbox.x;
  int left_y = face_loc.bbox.y;
  int bbox_w = face_loc.bbox.width;
  int bbox_h = face_loc.bbox.height;
  int right_x = left_x + bbox_w - 1;
  int right_y = left_y + bbox_h - 1;

  float extend_factor = 0.05;
  float extend_revised_y = 0.05;

  /*Compute the extended region of the detected face*/
  int extend_lx = std::max(int(floor(left_x - extend_factor*bbox_w)), int(0));
  int extend_rx = std::min(int(floor(right_x + extend_factor*bbox_w)), int(im_width - 1));
  int extend_ly = std::max(int(floor(left_y - (extend_factor - extend_revised_y)*bbox_h)), int(0));
  int extend_ry = std::min(int(floor(right_y + (extend_factor + extend_revised_y)*bbox_h)), int(im_height - 1));

  int face_w = extend_rx - extend_lx + 1;
  int face_h = extend_ry - extend_ly + 1;

  /*Get the face image based on the extended face region*/
  unsigned char *face_patch = new unsigned char[face_w*face_h];
  for (int h = 0; h < face_h; h++)
  {
    const unsigned char *p_origin = gray_im + (h + extend_ly)*im_width + extend_lx;
    unsigned char *p_dest = face_patch + h*face_w;
    memcpy(p_dest, p_origin, face_w);
  }

  /*The first local stacked autoencoder network*/
  double *fea = new double[fea_dim_];
  int lan1_resize_w = 80;
  int lan1_resize_h = 80;
  BYTE *lan1_patch = new BYTE[lan1_resize_w*lan1_resize_h];
  ResizeImage(face_patch, face_w, face_h, lan1_patch, lan1_resize_w, lan1_resize_h);

  for (int i = 0; i < pts_num_; i++)
  {
    facial_loc[i * 2] = mean_shape_[i * 2] - 1;
    facial_loc[i * 2 + 1] = mean_shape_[i * 2 + 1] - 1;
  }

  /*Extract the shape indexed SIFT features*/
  TtSift(lan1_patch, lan1_resize_w, lan1_resize_h, facial_loc, 32, fea);

  float *re_fea = new float[fea_dim_];
  for (int i = 0; i < 128; i++)
  {
    for (int j = 0; j < pts_num_; j++)
    {
      if (isnan(fea[j * 128 + i]))
      {
        re_fea[i*pts_num_ + j] = 0;
      }
      else
      {
        re_fea[i*pts_num_ + j] = fea[j * 128 + i];
      }
    }
  }

  float ** lan1_a = new float *[lan1_size_];

  for (int i = 0; i < lan1_size_; i++)
  {
    lan1_a[i] = new float[lan1_structure_[i]];
  }

  for (int i = 0; i < fea_dim_; i++)
  {
    lan1_a[0][i] = re_fea[i];
  }

  for (int i = 0; i < lan1_size_ - 1; i++)
  {
    for (int j = 0; j < lan1_structure_[i + 1]; j++)
    {
      float inner_product = 0;
      int fea_dim = lan1_structure_[i];
      for (int k = 0; k < fea_dim; k++)
      {
        inner_product = inner_product + lan1_a[i][k] * lan1_w_[i][j*fea_dim + k];
      }
      if (i == lan1_size_ - 2)
      {
        lan1_a[i + 1][j] = inner_product + lan1_b_[i][j];
      }
      else
      {
        lan1_a[i + 1][j] = 1.0 / (1 + exp(-inner_product - lan1_b_[i][j]));
      }

    }
  }
  for (int i = 0; i < pts_num_ * 2; i++)
  {
    facial_loc[i] = facial_loc[i] + lan1_a[lan1_size_ - 1][i];
  }
  for (int i = 0; i < lan1_size_; i++)
  {
    delete[](lan1_a[i]);
  }
  delete[]lan1_a;
  delete[]lan1_patch;

  /*The second local stacked autoencoder network*/
  int lan2_resize_w = 140;
  int lan2_resize_h = 140;
  BYTE *lan2_patch = new BYTE[lan2_resize_w*lan2_resize_h];
  ResizeImage(face_patch, face_w, face_h, lan2_patch, lan2_resize_w, lan2_resize_h);

  float x_scale = float(lan1_resize_w) / lan2_resize_w;
  float y_scale = float(lan1_resize_h) / lan2_resize_h;

  for (int i = 0; i < pts_num_; i++)
  {
    facial_loc[i * 2] = (facial_loc[i * 2]) / x_scale;
    facial_loc[i * 2 + 1] = (facial_loc[i * 2 + 1]) / y_scale;
  }
  /*Extract the shape indexed SIFT features*/
  TtSift(lan2_patch, lan2_resize_w, lan2_resize_h, facial_loc, 32, fea);

  for (int i = 0; i < 128; i++)
  {
    for (int j = 0; j < pts_num_; j++)
    {
      if (isnan(fea[j * 128 + i]))
      {
        re_fea[i*pts_num_ + j] = 0;
      }
      else
      {
        re_fea[i*pts_num_ + j] = fea[j * 128 + i];
      }
    }
  }

  float ** lan2_a = new float *[lan2_size_];

  for (int i = 0; i < lan2_size_; i++)
  {
    lan2_a[i] = new float[lan2_structure_[i]];
  }

  for (int i = 0; i < fea_dim_; i++)
  {
    lan2_a[0][i] = re_fea[i];
  }

  for (int i = 0; i < lan2_size_ - 1; i++)
  {
    for (int j = 0; j < lan2_structure_[i + 1]; j++)
    {
      float inner_product = 0;
      int fea_dim = lan2_structure_[i];
      for (int k = 0; k < fea_dim; k++)
      {
        inner_product = inner_product + lan2_a[i][k] * lan2_w_[i][j*fea_dim + k];
      }
      if (i == lan2_size_ - 2)
      {
        lan2_a[i + 1][j] = inner_product + lan2_b_[i][j];
      }
      else
      {
        lan2_a[i + 1][j] = 1.0 / (1 + exp(-inner_product - lan2_b_[i][j]));
      }

    }
  }
  for (int i = 0; i < pts_num_ * 2; i++)
  {
    facial_loc[i] = facial_loc[i] + lan2_a[lan2_size_ - 1][i];
  }
  for (int i = 0; i < lan2_size_; i++)
  {
    delete[](lan2_a[i]);
  }
  delete[]lan2_a;
  delete[]lan2_patch;

  delete[]fea;
  delete[]re_fea;
  delete[]face_patch;

  x_scale = float(lan2_resize_w) / face_w;
  y_scale = float(lan2_resize_h) / face_h;

  for (int i = 0; i < pts_num_; i++)
  {
    facial_loc[i * 2] = (facial_loc[i * 2]) / x_scale + extend_lx;
    facial_loc[i * 2 + 1] = (facial_loc[i * 2 + 1]) / y_scale + extend_ly;
  }
}

/** Extract shape indexed SIFT features.
  *  @param gray_im A grayscale image
  *  @param im_width The width of the inpute image
  *  @param im_height The height of the inpute image
  *  @param face_shape The locations of facial points
  *  @param patch_size The size of the patch used for extracting SIFT feature
  *  @param[out] sift_fea the extracted shape indexed SIFT features which are concatenated into a vector
  */
void CCFAN::TtSift(const unsigned char *gray_im, int im_width, int im_height, float *face_shape, int patch_size, double *sift_fea)
{
  unsigned char *sub_img = new unsigned char[patch_size*patch_size];
  SIFT* sift_extractor = new SIFT();
  sift_extractor->InitSIFT(patch_size, patch_size, 32, 16);
  double *one_sift_fea = new double[128];
  double *fea_header = sift_fea;

  for (int i = 0; i < pts_num_; i++)
  {
    /*Get one image patch*/
    GetSubImg(gray_im, im_width, im_height, face_shape[i * 2], face_shape[i * 2 + 1], patch_size, sub_img);
    /*Extract  one SIFT feature of one image patch*/
    sift_extractor->CalcSIFT(sub_img, one_sift_fea);
    memcpy(fea_header + i * 128, one_sift_fea, 128 * 8);
  }
  delete[]one_sift_fea;
  delete[]sub_img;
  delete sift_extractor;
}

/** Extract a image patch which is centered at point(point_x, point_y) with a given patch size.
  *  @param gray_im A grayscale image
  *  @param im_width The width of the inpute image
  *  @param im_height The height of the inpute image
  *  @param point_x The X coordinate of one point
  *  @param point_y The Y coordinate of one point
  *  @param patch_size The size of the extracted patch
  *  @param[out] sub_img A grayscale image patch
  */
void CCFAN::GetSubImg(const unsigned char *gray_im, int im_width, int im_height, float point_x, float point_y, int patch_size, BYTE *sub_img)
{
  memset(sub_img, 128, patch_size*patch_size);
  int center_x = floor(point_x + 0.5);
  int center_y = floor(point_y + 0.5);
  int patch_left = std::max((center_x + 1) - patch_size / 2, 0);
  int patch_right = std::min((center_x + 1) + patch_size / 2 - 1, im_width - 1);
  int patch_top = std::max((center_y + 1) - patch_size / 2, 0);
  int patch_bottom = std::min((center_y + 1) + patch_size / 2 - 1, im_height - 1);

  int lx = abs(patch_left - ((center_x + 1) - patch_size / 2));
  int rx = patch_size - abs(patch_right - ((center_x + 1) + patch_size / 2 - 1)) - 1;
  int ty = abs(patch_top - ((center_y + 1) - patch_size / 2));
  int by = patch_size - abs(patch_bottom - ((center_y + 1) + patch_size / 2 - 1)) - 1;

  for (int h = ty, ph = patch_top; h < by + 1; h++, ph++)
  {
    for (int w = lx, pw = patch_left; w < rx + 1; w++, pw++)
    {
      sub_img[h*patch_size + w] = gray_im[ph*im_width + pw];
    }
  }
}

/** Resize the image by bilinear interpolation.
  *  @param src_im A source image in grayscale
  *  @param src_width The width of the source image
  *  @param src_height The height of the source image
  *  @param[out] dst_im The target image in grayscale
  *  @param dst_width The width of the target image
  *  @param dst_height The height of the target image
  */
bool CCFAN::ResizeImage(const unsigned char *src_im, int src_width, int src_height,
  unsigned char* dst_im, int dst_width, int dst_height)
{

  double	lfx_scl, lfy_scl;
  if (src_width == dst_width && src_height == dst_height) {
    memcpy(dst_im, src_im, src_width * src_height * sizeof(unsigned char));
    return true;
  }

  lfx_scl = double(src_width + 0.0) / dst_width;
  lfy_scl = double(src_height + 0.0) / dst_height;

  for (int n_y_d = 0; n_y_d < dst_height; n_y_d++) {
    for (int n_x_d = 0; n_x_d < dst_width; n_x_d++) {
      double lf_x_s = lfx_scl * n_x_d;
      double lf_y_s = lfy_scl * n_y_d;

      int n_x_s = int(lf_x_s);
      n_x_s = (n_x_s <= (src_width - 2) ? n_x_s : (src_width - 2));
      int n_y_s = int(lf_y_s);
      n_y_s = (n_y_s <= (src_height - 2) ? n_y_s : (src_height - 2));

      double lf_weight_x = lf_x_s - n_x_s;
      double lf_weight_y = lf_y_s - n_y_s;

      double lf_new_gray = (1 - lf_weight_y) * ((1 - lf_weight_x) * src_im[n_y_s * src_width + n_x_s] +
        lf_weight_x * src_im[n_y_s * src_width + n_x_s + 1]) +
        lf_weight_y * ((1 - lf_weight_x) * src_im[(n_y_s + 1) * src_width + n_x_s] +
        lf_weight_x * src_im[(n_y_s + 1) * src_width + n_x_s + 1]);

      dst_im[n_y_d * dst_width + n_x_d] = (unsigned char)(lf_new_gray);
    }
  }
  return true;
}


