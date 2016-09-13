## SeetaFace Identification

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](../LICENSE)

### Brief Description

As most state-of-the-art face recognition technologies, SeetaFace Identification is based on deep convolutional neural network (DCNN). Specifically, it is an implementation of the VIPLFaceNet, which consists of 7 convolutional layers and 2 fully-connected layers with input size of 256x256x3. Simply speaking, it is tailored from AlexNet for fast and accurate face recognition in the wild based on the cutting-edge findings in deep learning research. Compared with AlexNet, VIPLFaceNet is deeper by replacing the 5x5 convolutional kernel layer with two 3x3 kernel layers, and it also reduces the number of kernels in each layer. Additionally, VIPLFaceNet introduces the Fast Normalization Layer for faster convergence of the network optimization. Evaluation shows that, compared with original AlexNet, the error rate of VIPLFaceNet is reduced by 40% in case of the same training set, while its training time is reduced by 80% and its computational cost for feature extraction is reduced by 40%. 

The model shipped in with the codes is trained with 1.4M face images of 16K subjects including both Mongolians and Caucasians. In the SeetaFace open-source face identification engine, the outputs of the 2048 nodes of the FC2 layer in the VIPLFaceNet are exploited as the feature of the input face. For verification or identification, the cosine similarity between their 2048D features of two given faces is used for thresholding or ranking. SeetaFace identification achieves a mean accuracy of 97.1% on LFW under the standard Image-Restricted protocol. To reproduce this result, one needs to use SeetaFace Detection and SeetaFace Alignment for face detection and alignment respectively. Please note that, for few faces not detected, one should use the central region of the original images instead for face alignment. In terms of speed, when tested on a single core of i7-3770 CPU, the engine can extract the features of a face in 115ms (excluding face detection and alignment). 


### Compile
    
#### Linux & Mac OS

Change current working directory to `SeetaFace/FaceIdentification` and run the following command:

``` 
mkdir build
cd build
cmake .. && make
```

If everything goes fine, move on to test the program:
```
./build/src/test/test_face_recognizer.bin
```

#### Windows

A Visual Studio 2013 solution is provided in the subdirectory [**examples**](./examples). The solution contains 2 projects:
(1) [**Identification**](./examples/Identification) which generates the shared lib, and (2) [**Tester**](./examples/Tester)
which performs face recognition using the generated lib. To build the Tester project, one needs to ensure the configuration
of [OpenCV](http://opencv.org/) is correctly set.

### How to Use SeetaFace Identification

The class for face identification is included in `seeta` namespace. To use the 
function of SeetaFace Identification, one should first instantiate an object of 
`seeta::FaceIdentification` with path of the model file. The model file seeta_fr_v1.0.bin can be achieved by unzipping the files "seeta_fr_v1.0.part1.rar" and "seeta_fr_v1.0.part2.rar" in subdirectory [**model**](./model).  

```c++
FaceIdentification face_recognizer("seeta_fr_v1.0.bin");
```

After a face image is read, one needs to pack the image data with `seeta::ImageData`. 
Note that the pixel values should stored in a continuous 1D array in row-major 
style.

```c++
seeta::ImageData img_data(width, height, channels);
img_data.data = img_data_buf;
```

To get features of the face, call `ExtractFeatureWithCrop()` after obtaining the coordiates of the
5 landmarks (`pt5`) with [SeetaFace Alignment](https://github.com/seetaface/SeetaFaceEngine/tree/master/FaceAlignment).

```c++
float* feats = new float[face_recognizer.feature_size()];
FacialLandmark pt5[5];
face_recognizer.ExtractFeatureWithCrop(img_data, pt5, feats);
```

To calculate the similarity between two faces, call `CalcSimilarity()` with the corresponding feature vectors.

```c++
float sim = face_recognizer.CalcSimilarity(feats1, feats2);
```
See an [example test file](./src/test/test_face_recognizer.cpp) for details.

### Citation

If you find SeetaFace Identification (VIPLFaceNet) useful in your research work, please consider citing:

    @article{liu2016viplfacenet,
      Author = {Liu, Xin, and Kan, Meina, and Wu, Wanglong, and Shan, Shiguang, and Chen, Xilin},
      Journal = {Frontiers of Computer Science},
      Title = {{VIPLFaceNet}: An Open Source Deep Face Recognition SDK},
      Year = {2016}
    }

### License

SeetaFace Identification is released under the [BSD 2-Clause license](../LICENSE).
