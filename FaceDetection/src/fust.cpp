/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Detection module, containing codes implementing the
 * face detection method described in the following paper:
 *
 *
 *   Funnel-structured cascade for multi-view face detection with alignment awareness,
 *   Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
 *   In Neurocomputing (under review)
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Shuzhe Wu (a Ph.D supervised by Prof. Shiguang Shan)
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

#include "fust.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "classifier/lab_boosted_classifier.h"
#include "classifier/surf_mlp.h"
#include "feat/lab_feature_map.h"
#include "feat/surf_feature_map.h"
#include "io/lab_boost_model_reader.h"
#include "io/surf_mlp_model_reader.h"
#include "util/nms.h"

namespace seeta {
namespace fd {

bool FuStDetector::LoadModel(const std::string & model_path) {
  std::ifstream model_file(model_path, std::ifstream::binary);
  bool is_loaded = true;

  if (!model_file.is_open()) {
    is_loaded = false;
  } else {
    hierarchy_size_.clear();
    num_stage_.clear();
    wnd_src_id_.clear();

    int32_t hierarchy_size;
    int32_t num_stage;
    int32_t num_wnd_src;
    int32_t type_id;
    int32_t feat_map_index = 0;
    std::shared_ptr<seeta::fd::ModelReader> reader;
    std::shared_ptr<seeta::fd::Classifier> classifier;
    seeta::fd::ClassifierType classifier_type;

    model_file.read(reinterpret_cast<char*>(&num_hierarchy_), sizeof(int32_t));
    for (int32_t i = 0; is_loaded && i < num_hierarchy_; i++) {
      model_file.read(reinterpret_cast<char*>(&hierarchy_size),
        sizeof(int32_t));
      hierarchy_size_.push_back(hierarchy_size);

      for (int32_t j = 0; is_loaded && j < hierarchy_size; j++) {
        model_file.read(reinterpret_cast<char*>(&num_stage), sizeof(int32_t));
        num_stage_.push_back(num_stage);

        for (int32_t k = 0; is_loaded && k < num_stage; k++) {
          model_file.read(reinterpret_cast<char*>(&type_id), sizeof(int32_t));
          classifier_type = static_cast<seeta::fd::ClassifierType>(type_id);
          reader = CreateModelReader(classifier_type);
          classifier = CreateClassifier(classifier_type);

          is_loaded = !model_file.fail() &&
            reader->Read(&model_file, classifier.get());
          if (is_loaded) {
            model_.push_back(classifier);
            std::shared_ptr<seeta::fd::FeatureMap> feat_map;
            if (cls2feat_idx_.count(classifier_type) == 0) {
              feat_map_.push_back(CreateFeatureMap(classifier_type));
              cls2feat_idx_.insert(
                std::map<seeta::fd::ClassifierType, int32_t>::value_type(
                classifier_type, feat_map_index++));
            }
            feat_map = feat_map_[cls2feat_idx_.at(classifier_type)];
            model_.back()->SetFeatureMap(feat_map.get());
          }
        }

        wnd_src_id_.push_back(std::vector<int32_t>());
        model_file.read(reinterpret_cast<char*>(&num_wnd_src), sizeof(int32_t));
        if (num_wnd_src > 0) {
          wnd_src_id_.back().resize(num_wnd_src);
          for (int32_t k = 0; k < num_wnd_src; k++) {
            model_file.read(reinterpret_cast<char*>(&(wnd_src_id_.back()[k])),
              sizeof(int32_t));
          }
        }
      }
    }

    model_file.close();
  }

  return is_loaded;
}

std::vector<seeta::FaceInfo> FuStDetector::Detect(
    seeta::fd::ImagePyramid* img_pyramid) {
  float score;
  seeta::FaceInfo wnd_info;
  seeta::Rect wnd;
  float scale_factor = 0.0;
  const seeta::ImageData* img_scaled =
    img_pyramid->GetNextScaleImage(&scale_factor);

  wnd.height = wnd.width = wnd_size_;

  // Sliding window

  std::vector<std::vector<seeta::FaceInfo> > proposals(hierarchy_size_[0]);
  std::shared_ptr<seeta::fd::FeatureMap> & feat_map_1 =
    feat_map_[cls2feat_idx_[model_[0]->type()]];

  while (img_scaled != nullptr) {
    feat_map_1->Compute(img_scaled->data, img_scaled->width,
      img_scaled->height);

    wnd_info.bbox.width = static_cast<int32_t>(wnd_size_ / scale_factor + 0.5);
    wnd_info.bbox.height = wnd_info.bbox.width;

    int32_t max_x = img_scaled->width - wnd_size_;
    int32_t max_y = img_scaled->height - wnd_size_;
    for (int32_t y = 0; y <= max_y; y += slide_wnd_step_y_) {
      wnd.y = y;
      for (int32_t x = 0; x <= max_x; x += slide_wnd_step_x_) {
        wnd.x = x;
        feat_map_1->SetROI(wnd);

        wnd_info.bbox.x = static_cast<int32_t>(x / scale_factor + 0.5);
        wnd_info.bbox.y = static_cast<int32_t>(y / scale_factor + 0.5);

        for (int32_t i = 0; i < hierarchy_size_[0]; i++) {
          if (model_[i]->Classify(&score)) {
            wnd_info.score = static_cast<double>(score);
            proposals[i].push_back(wnd_info);
          }
        }
      }
    }

    img_scaled = img_pyramid->GetNextScaleImage(&scale_factor);
  }

  std::vector<std::vector<seeta::FaceInfo> > proposals_nms(hierarchy_size_[0]);
  for (int32_t i = 0; i < hierarchy_size_[0]; i++) {
    seeta::fd::NonMaximumSuppression(&(proposals[i]),
      &(proposals_nms[i]), 0.8f);
    proposals[i].clear();
  }

  // Following classifiers

  seeta::ImageData img = img_pyramid->image1x();
  seeta::Rect roi;
  std::vector<float> mlp_predicts(4);  // @todo no hard-coded number!
  roi.x = roi.y = 0;
  roi.width = roi.height = wnd_size_;

  int32_t cls_idx = hierarchy_size_[0];
  int32_t model_idx = hierarchy_size_[0];
  std::vector<int32_t> buf_idx;

  for (int32_t i = 1; i < num_hierarchy_; i++) {
    buf_idx.resize(hierarchy_size_[i]);
    for (int32_t j = 0; j < hierarchy_size_[i]; j++) {
      int32_t num_wnd_src = static_cast<int32_t>(wnd_src_id_[cls_idx].size());
      std::vector<int32_t> & wnd_src = wnd_src_id_[cls_idx];
      buf_idx[j] = wnd_src[0];
      proposals[buf_idx[j]].clear();
      for (int32_t k = 0; k < num_wnd_src; k++) {
        proposals[buf_idx[j]].insert(proposals[buf_idx[j]].end(),
          proposals_nms[wnd_src[k]].begin(), proposals_nms[wnd_src[k]].end());
      }

      std::shared_ptr<seeta::fd::FeatureMap> & feat_map =
        feat_map_[cls2feat_idx_[model_[model_idx]->type()]];
      for (int32_t k = 0; k < num_stage_[cls_idx]; k++) {
        int32_t num_wnd = static_cast<int32_t>(proposals[buf_idx[j]].size());
        std::vector<seeta::FaceInfo> & bboxes = proposals[buf_idx[j]];
        int32_t bbox_idx = 0;

        for (int32_t m = 0; m < num_wnd; m++) {
          if (bboxes[m].bbox.x + bboxes[m].bbox.width <= 0 ||
              bboxes[m].bbox.y + bboxes[m].bbox.height <= 0)
            continue;
          GetWindowData(img, bboxes[m].bbox);
          feat_map->Compute(wnd_data_.data(), wnd_size_, wnd_size_);
          feat_map->SetROI(roi);

          if (model_[model_idx]->Classify(&score, mlp_predicts.data())) {
            float x = static_cast<float>(bboxes[m].bbox.x);
            float y = static_cast<float>(bboxes[m].bbox.y);
            float w = static_cast<float>(bboxes[m].bbox.width);
            float h = static_cast<float>(bboxes[m].bbox.height);

            bboxes[bbox_idx].bbox.width =
              static_cast<int32_t>((mlp_predicts[3] * 2 - 1) * w + w + 0.5);
            bboxes[bbox_idx].bbox.height = bboxes[bbox_idx].bbox.width;
            bboxes[bbox_idx].bbox.x =
              static_cast<int32_t>((mlp_predicts[1] * 2 - 1) * w + x +
              (w - bboxes[bbox_idx].bbox.width) * 0.5 + 0.5);
            bboxes[bbox_idx].bbox.y =
              static_cast<int32_t>((mlp_predicts[2] * 2 - 1) * h + y +
              (h - bboxes[bbox_idx].bbox.height) * 0.5 + 0.5);
            bboxes[bbox_idx].score = score;
            bbox_idx++;
          }
        }
        proposals[buf_idx[j]].resize(bbox_idx);

        if (k < num_stage_[cls_idx] - 1) {
          seeta::fd::NonMaximumSuppression(&(proposals[buf_idx[j]]),
            &(proposals_nms[buf_idx[j]]), 0.8f);
          proposals[buf_idx[j]] = proposals_nms[buf_idx[j]];
        } else {
          if (i == num_hierarchy_ - 1) {
            seeta::fd::NonMaximumSuppression(&(proposals[buf_idx[j]]),
              &(proposals_nms[buf_idx[j]]), 0.3f);
            proposals[buf_idx[j]] = proposals_nms[buf_idx[j]];
          }
        }
        model_idx++;
      }

      cls_idx++;
    }

    for (int32_t j = 0; j < hierarchy_size_[i]; j++)
      proposals_nms[j] = proposals[buf_idx[j]];
  }

  return proposals_nms[0];
}

std::shared_ptr<seeta::fd::ModelReader>
FuStDetector::CreateModelReader(seeta::fd::ClassifierType type) {
  std::shared_ptr<seeta::fd::ModelReader> reader;
  switch (type) {
  case seeta::fd::ClassifierType::LAB_Boosted_Classifier:
    reader.reset(new seeta::fd::LABBoostModelReader());
    break;
  case seeta::fd::ClassifierType::SURF_MLP:
    reader.reset(new seeta::fd::SURFMLPModelReader());
    break;
  default:
    break;
  }
  return reader;
}

std::shared_ptr<seeta::fd::Classifier>
FuStDetector::CreateClassifier(seeta::fd::ClassifierType type) {
  std::shared_ptr<seeta::fd::Classifier> classifier;
  switch (type) {
  case seeta::fd::ClassifierType::LAB_Boosted_Classifier:
    classifier.reset(new seeta::fd::LABBoostedClassifier());
    break;
  case seeta::fd::ClassifierType::SURF_MLP:
    classifier.reset(new seeta::fd::SURFMLP());
    break;
  default:
    break;
  }
  return classifier;
}

std::shared_ptr<seeta::fd::FeatureMap>
FuStDetector::CreateFeatureMap(seeta::fd::ClassifierType type) {
  std::shared_ptr<seeta::fd::FeatureMap> feat_map;
  switch (type) {
  case seeta::fd::ClassifierType::LAB_Boosted_Classifier:
    feat_map.reset(new seeta::fd::LABFeatureMap());
    break;
  case seeta::fd::ClassifierType::SURF_MLP:
    feat_map.reset(new seeta::fd::SURFFeatureMap());
    break;
  default:
    break;
  }
  return feat_map;
}

void FuStDetector::GetWindowData(const seeta::ImageData & img,
    const seeta::Rect & wnd) {
  int32_t pad_left;
  int32_t pad_right;
  int32_t pad_top;
  int32_t pad_bottom;
  seeta::Rect roi = wnd;

  pad_left = pad_right = pad_top = pad_bottom = 0;
  if (roi.x + roi.width > img.width)
    pad_right = roi.x + roi.width - img.width;
  if (roi.x < 0) {
    pad_left = -roi.x;
    roi.x = 0;
  }
  if (roi.y + roi.height > img.height)
    pad_bottom = roi.y + roi.height - img.height;
  if (roi.y < 0) {
    pad_top = -roi.y;
    roi.y = 0;
  }

  wnd_data_buf_.resize(roi.width * roi.height);
  const uint8_t* src = img.data + roi.y * img.width + roi.x;
  uint8_t* dest = wnd_data_buf_.data();
  int32_t len = sizeof(uint8_t) * roi.width;
  int32_t len2 = sizeof(uint8_t) * (roi.width - pad_left - pad_right);

  if (pad_top > 0) {
    std::memset(dest, 0, len * pad_top);
    dest += (roi.width * pad_top);
  }
  if (pad_left == 0) {
    if (pad_right == 0) {
      for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
        std::memcpy(dest, src, len);
        src += img.width;
        dest += roi.width;
      }
    } else {
      for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
        std::memcpy(dest, src, len2);
        src += img.width;
        dest += roi.width;
        std::memset(dest - pad_right, 0, sizeof(uint8_t) * pad_right);
      }
    }
  } else {
    if (pad_right == 0) {
      for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
        std::memset(dest, 0, sizeof(uint8_t)* pad_left);
        std::memcpy(dest + pad_left, src, len2);
        src += img.width;
        dest += roi.width;
      }
    } else {
      for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
        std::memset(dest, 0, sizeof(uint8_t) * pad_left);
        std::memcpy(dest + pad_left, src, len2);
        src += img.width;
        dest += roi.width;
        std::memset(dest - pad_right, 0, sizeof(uint8_t) * pad_right);
      }
    }
  }
  if (pad_bottom > 0)
    std::memset(dest, 0, len * pad_bottom);

  seeta::ImageData src_img(roi.width, roi.height);
  seeta::ImageData dest_img(wnd_size_, wnd_size_);
  src_img.data = wnd_data_buf_.data();
  dest_img.data = wnd_data_.data();
  seeta::fd::ResizeImage(src_img, &dest_img);
}

}  // namespace fd
}  // namespace seeta
