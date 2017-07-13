/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is an example of how to use SeetaFace engine for face alignment, the
 * face alignment method described in the following paper:
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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#ifdef _WIN32
#pragma once
#include <opencv2/core/version.hpp>

#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) \
  CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif //_DEBUG

#pragma comment( lib, cvLIB("core") )
#pragma comment( lib, cvLIB("imgproc") )
#pragma comment( lib, cvLIB("highgui") )

#endif //_WIN32

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "face_detection.h"
#include "face_alignment.h"

#ifdef _WIN32
std::string MODEL_DIR = "../../FaceDetection/model/";
std::string ALIENMENT_MODEL_DIR = "../../FaceAlignment/model/";
#else
std::string MODEL_DIR = "../../../FaceDetection/model/";
std::string ALIENMENT_MODEL_DIR = "../../../FaceAlignment/model/";
#endif

#define PI 3.14159265

int main(int argc, char** argv)
{
  const char* img_path = "test_image.jpg";
  if (argc >= 2) {
    img_path = argv[1];
  }

  // Initialize face detection model
  seeta::FaceDetection detector((MODEL_DIR + "seeta_fd_frontal_v1.0.bin").c_str());
  detector.SetMinFaceSize(40);
  detector.SetScoreThresh(2.f);
  detector.SetImagePyramidScaleFactor(0.8f);
  detector.SetWindowStep(4, 4);

  // Initialize face alignment model 
  seeta::FaceAlignment point_detector((ALIENMENT_MODEL_DIR + "seeta_fa_v1.1.bin").c_str());

  //load image
  cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
  if (!img.data)
  {
    std::cerr << "Could not open or find the image" << std::endl;
    return -1;
  }

  cv::Mat img_gray;
  if (img.channels() != 1)
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
  else
    img_gray = img;

  int pts_num = 5;

  seeta::ImageData img_data;
  img_data.data = img_gray.data;
  img_data.width = img_gray.cols;
  img_data.height = img_gray.rows;
  img_data.num_channels = 1;

  // Detect faces
  std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);
  if (faces.empty()) {
    return -2;
  }

  for (auto face : faces) {
    // Detect 5 facial landmarks
    seeta::FacialLandmark points[5];
    point_detector.PointDetectLandmarks(img_data, face, points);

    // Visualize the results
    cv::RotatedRect rrect = cv::RotatedRect(cv::Point(face.bbox.x + face.bbox.width / 2, face.bbox.y + face.bbox.height / 2), cv::Size(face.bbox.width, face.bbox.height), atan((points[1].y - points[0].y) / (points[1].x - points[0].x)) * 180 / PI);
    cv::Point2f vertices[4];
    rrect.points(vertices);
    for (int i = 0; i < 4; i++)
      cv::line(img, vertices[i], vertices[(i + 1) % 4], CV_RGB(255, 0, 0), 2, CV_AA);

    for (int i = 0; i < pts_num; i++)
    {
      cv::circle(img, cvPoint(points[i].x, points[i].y), 2, CV_RGB(0, 255, 0), CV_FILLED);
    }
  }
  
  // Show crop face
  cv::imshow("Alignment Face", img);
  cv::waitKey(0);
  cv::destroyWindow("Source Face");

  return 0;
}
