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

#ifndef SEETA_FD_CLASSIFIER_LAB_BOOSTED_CLASSIFIER_H_
#define SEETA_FD_CLASSIFIER_LAB_BOOSTED_CLASSIFIER_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "classifier.h"
#include "feat/lab_feature_map.h"

namespace seeta {
namespace fd {

/**
 * @class LABBaseClassifier
 * @brief Base classifier using LAB feature.
 */
class LABBaseClassifier {
 public:
  LABBaseClassifier()
    : num_bin_(255), thresh_(0.0f) {
    weights_.resize(num_bin_ + 1);
  }

  ~LABBaseClassifier() {}

  void SetWeights(const float* weights, int32_t num_bin);

  inline void SetThreshold(float thresh) { thresh_ = thresh; }

  inline int32_t num_bin() const { return num_bin_; }
  inline float weights(int32_t val) const { return weights_[val]; }
  inline float threshold() const { return thresh_; }

 private:
  int32_t num_bin_;

  std::vector<float> weights_;
  float thresh_;
};

/**
 * @class LABBoostedClassifier
 * @Brief A strong classifier constructed from base classifiers using LAB features.
 */
class LABBoostedClassifier : public Classifier {
 public:
  LABBoostedClassifier() : use_std_dev_(true) {}
  virtual ~LABBoostedClassifier() {}

  virtual bool Classify(float* score = nullptr, float* outputs = nullptr);

  inline virtual seeta::fd::ClassifierType type() {
    return seeta::fd::ClassifierType::LAB_Boosted_Classifier;
  }

  void AddFeature(int32_t x, int32_t y);
  void AddBaseClassifier(const float* weights, int32_t num_bin, float thresh);

  inline virtual void SetFeatureMap(seeta::fd::FeatureMap* featMap) {
    feat_map_ = dynamic_cast<seeta::fd::LABFeatureMap*>(featMap);
  }

  inline void SetUseStdDev(bool useStdDev) { use_std_dev_ = useStdDev; }

 private:
  static const int32_t kFeatGroupSize = 10;
  const float kStdDevThresh = 10.0f;

  std::vector<seeta::fd::LABFeature> feat_;
  std::vector<std::shared_ptr<seeta::fd::LABBaseClassifier> > base_classifiers_;
  seeta::fd::LABFeatureMap* feat_map_;
  bool use_std_dev_;
};

}  // namespace fd
}  // namespace seeta

#endif  // SEETA_FD_CLASSIFIER_LAB_BOOSTED_CLASSIFIER_H_
