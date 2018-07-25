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

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "face_detection.h"
#include "face_alignment.h"

int main(int argc, char** argv)
{
	if (argc < 4) {
		std::cout << "Usage: " << argv[0]
			<< " detection_model_path alignment_model_path image_path"
			<< std::endl;
		return -1;
	}

	// Initialize face detection model
	seeta::FaceDetection detector(argv[1]);
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	// Initialize face alignment model 
	seeta::FaceAlignment point_detector(argv[2]);

	//load image
	cv::Mat img = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
	cv::Mat img_gray;

	if (img.channels() != 1)
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	else
		img_gray = img;

	seeta::ImageData img_data;
	img_data.data = img_gray.data;
	img_data.width = img_gray.cols;
	img_data.height = img_gray.rows;
	img_data.num_channels = 1;

	// Detect faces
	std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);

	// Detect 5 facial landmarks
	for (int i = 0; i < faces.size(); i++)
	{
		cv::Rect face_rect;
		face_rect.x = faces[i].bbox.x;
		face_rect.y = faces[i].bbox.y;
		face_rect.width = faces[i].bbox.width;
		face_rect.height = faces[i].bbox.height;
		cv::rectangle(img, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);

		// Visualize the results
		seeta::FacialLandmark points[5];
		if (point_detector.PointDetectLandmarks(img_data, faces[0], points))
		{
			for (int j = 0; j < 5; j++)
				cv::circle(img, cv::Point(points[j].x, points[j].y),
					2, CV_RGB(0, 255, 0), CV_FILLED);
		}
	}

	cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
	cv::imshow("Test", img);
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}
