LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE_TAGS := optional
LOCAL_MODULE := FaceIdentification

DIRS = src tools

LOCAL_SRC_FILES := FaceIdentification/src/blob.cpp \
	FaceIdentification/src/common_net.cpp \
	FaceIdentification/src/bias_adder_net.cpp \
	FaceIdentification/src/bn_net.cpp \
	FaceIdentification/src/conv_net.cpp \
	FaceIdentification/src/eltwise_net.cpp \
	FaceIdentification/src/inner_product_net.cpp \
	FaceIdentification/src/max_pooling_net.cpp \
	FaceIdentification/src/pad_net.cpp \
	FaceIdentification/src/spatial_transform_net.cpp \
	FaceIdentification/src/tform_maker_net.cpp \
	FaceIdentification/src/net.cpp \
	FaceIdentification/src/math_functions.cpp \
	FaceIdentification/src/log.cpp \
	FaceIdentification/tools/aligner.cpp \
	FaceIdentification/tools/face_identification.cpp

LOCAL_C_INCLUDES := $(LOCAL_PATH)/FaceIdentification/include \
	$(LOCAL_PATH)/FaceIdentification/src \
	$(LOCAL_PATH)/FaceIdentification/tools

LOCAL_CFLAGS := -mfloat-abi=softfp -mfpu=neon-vfpv4 -march=armv7-a
LOCAL_ARM_NEON := true
TARGET_ARCH_ABI := armeabi-v7a
LOCAL_ARM_MODE := arm

include $(BUILD_STATIC_LIBRARY)
