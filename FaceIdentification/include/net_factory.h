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

#ifndef NET_FACTORY_H_
#define NET_FACTORY_H_

#include "net.h"

#include <map>
#include <string>
#include <iostream>
#include <memory>

class NetRegistry {
 public:
  typedef std::shared_ptr<Net> (*Creator)();
  typedef std::map<std::string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  static void AddCreator(const std::string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    if (registry.count(type) != 0) {
      LOG(INFO) << "Net type " << type << " already registered.";
    }
    registry[type] = creator;
  }

  static std::shared_ptr<Net> CreateNet(const std::string type) {
    CreatorRegistry& registry = Registry();
    if (registry.count(type) != 1) {
      LOG(ERROR) << "Net type " << type << " haven't registered.";
    }
    return registry[type]();
  }
 private:
  NetRegistry() {}

};

class NetRegisterer {
 public:
  NetRegisterer(const std::string& type,
                std::shared_ptr<Net> (*creator)()) {
    LOG(INFO) << "Registering net type: " << type;
    NetRegistry::AddCreator(type, creator);
  }
};

#define REGISTER_NET_CREATOR(type, creator)                                    \
  static NetRegisterer g_creator_##type(#type, creator)

#define REGISTER_NET_CLASS(type)                                               \
  std::shared_ptr<Net> Creator_##type##Net()                                   \
  {                                                                            \
    return std::shared_ptr<Net>(new type##Net());                              \
  }                                                                            \
  REGISTER_NET_CREATOR(type, Creator_##type##Net);                             \
  static type##Net type
#endif //NET_FACTORY_H_
