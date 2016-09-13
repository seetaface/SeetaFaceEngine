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

#include "log.h"
#ifdef _WIN32
#include "windows.h"
#endif

ViplLog::~ViplLog() {
#ifdef __VIPL_LOG__
#ifdef _WIN32
  std::string out_str = my_cout_.str();
  std::string sys_cmd = "color 0";
  std::string log_color_str = "LOG_COLOR_";
  std::size_t s_idx = 0, p = out_str.find(log_color_str);
  std::size_t log_color_str_size = log_color_str.size();
  while (p != std::string::npos) {
    std::cout << out_str.substr(s_idx, p - s_idx);
    HANDLE hdl = GetStdHandle(STD_OUTPUT_HANDLE);
    WORD f_color;
    if (out_str[p + log_color_str_size] == 'R') {
      f_color = FOREGROUND_RED;
    }
    else if (out_str[p + log_color_str_size] == 'G') {
      f_color = FOREGROUND_GREEN;
    }
    else if (out_str[p + log_color_str_size] == 'W') {
      f_color = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED;
    }
    else if (out_str[p + log_color_str_size] == 'Y') {
      f_color = FOREGROUND_RED | FOREGROUND_GREEN;
    }
    SetConsoleTextAttribute(hdl, f_color | FOREGROUND_INTENSITY);
    s_idx = p + log_color_str_size + 1;
    p = out_str.find(log_color_str, s_idx);
  }
  std::cout << out_str.substr(s_idx) << std::endl;
#else
  std::cout << my_cout_.str() << std::endl;
#endif // _WIN32
#endif // __VIPL_LOG__
}
