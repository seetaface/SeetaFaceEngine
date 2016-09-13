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

#ifndef LOG_H_
#define LOG_H_

#include "viplnet.h"

#include <iostream>
#include <sstream>
#include <string>
#include <string.h>
#include <math.h>

#if defined(__unix__) || defined(__APPLE__)
#define _BEGIN_INFO_ "\033[1;32m"
#define _BEGIN_ERROR_ "\033[1;31m"
#define _BEGIN_DEBUG_ "\033[1;33m"
#define _END_COLOR_ "\033[0m"

#define ERROR _BEGIN_ERROR_ "[ERROR] "
#define DEBUG _BEGIN_DEBUG_ "[DEBUG] "
#define INFO _BEGIN_INFO_ "[INFO ] "
#define __FILENAME__ (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1)    \
:__FILE__)
#else
#define _BEGIN_INFO_ "LOG_COLOR_G"
#define _BEGIN_ERROR_ "LOG_COLOR_R"
#define _BEGIN_DEBUG_ "LOG_COLOR_Y"
#define _END_COLOR_ "LOG_COLOR_W"

#define ERROR _BEGIN_ERROR_ "[ERROR] "
#define DEBUG _BEGIN_DEBUG_ "[DEBUG] "
#define INFO _BEGIN_INFO_ "[INFO ] "
#define __FILENAME__ (strrchr(__FILE__, '\\') ? (strrchr(__FILE__, '\\') + 1)  \
:__FILE__)
#endif // __unix__, __APPLE__

class ViplLog {
 public:
  template<class T>
  ViplLog(const T &options) {
    my_cout_ << options;
  }
  ~ViplLog();
  template<class T>
  inline ViplLog &operator << (const T &x) {
#ifdef __VIPL_LOG__
    my_cout_ << x;
#endif
    return *this;
  }
 private:
  std::ostringstream my_cout_;
};

#define LOG(OPTION) ViplLog(OPTION) << "[" << __FILENAME__ << ":" <<           \
__LINE__ << "] " <<  _END_COLOR_

// check A == B
#define CHECK_EQ(A, B) if ((A) != (B)) LOG(ERROR) << "Check failed:" << "("    \
<< #A << " == " << #B  << ") " << "(" << A << " vs. " << B <<") " << "Inputs " \
<< #A<<" must be equal " << #B << "."

// check A != B
#define CHECK_NE(A, B) if ((A) == (B)) LOG(ERROR) << "Check failed:" << "("    \
<< #A << " != " << #B  << ") " << "(" << A << " vs. " << B <<") " << "Inputs " \
<< #A<<" must be not equal " << #B << "."

// check A < B
#define CHECK_LT(A, B) if ((A) >= (B)) LOG(ERROR) << "Check failed:" << "("    \
<< #A << " < " << #B  << ") " << "(" << A << " vs. " << B <<") " << "Inputs "  \
<< #A <<" must be less than " << #B <<"."

// check A > B
#define CHECK_GT(A, B) if ((A) <= (B)) LOG(ERROR) << "Check failed:" << "("    \
<< #A << " > " << #B << ") " << "(" << A << " vs. " << B <<") " << "Inputs "   \
<< #A << " must be great than " << #B << "."

// check A <= B
#define CHECK_LE(A, B) if ((A) > (B)) LOG(ERROR) << "Check failed:" << "("     \
 << #A << " <= " << #B  << ") " << "(" << A << " vs. " << B <<") " << "Inputs "\
<< #A <<" must be less than or equal to " << #B <<"."

// check A >= B
#define CHECK_GE(A, B) if ((A) < (B)) LOG(ERROR) << "Check failed:" << "("     \
<< #A << " >= " << #B << ") " << "(" << A << " vs. " << B <<") " << "Inputs "  \
<< #A << " must be great than or equal to " << #B << "."

// check int A is near to int B
#define CHECK_INT_NEAR(A, B, C) if (abs(A-B) > C) LOG(ERROR) << "Check failed:"\
<< #A << " is not near to " << #B << " within the range " << #C <<"."

// check double A is neart to double B
#define CHECK_DOUBLE_NEAR(A, B, C) if (fabs(A-B) > C) LOG(ERROR)               \
<< "Check failed: "  << #A << " is not near to " << #B << " within the range " \
<< #C <<"."

// check A is true
#define CHECK_TRUE(A) if (!(A)) LOG(ERROR) << "Check failed:" << "(" << #A     \
<< " Must be true" << ")."

#endif // LOG_H_
