LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE_TAGS := optional
LOCAL_MODULE := FaceDetection

DIRS = FaceDetection/src \
	FaceDetection/src/classifier \
	FaceDetection/src/feat \
	FaceDetection/src/io \
	FaceDetection/src/util \
	FaceDetection/include

INC  = $(foreach dir,$(DIRS),$(LOCAL_PATH)/$(dir))
SRC  = $(foreach dir,$(DIRS),$(wildcard $(LOCAL_PATH)/$(dir)/*.c))
SRC += $(foreach dir,$(DIRS),$(wildcard $(LOCAL_PATH)/$(dir)/*.cpp))

LOCAL_SRC_FILES := $(SRC:$(LOCAL_PATH)/%=%)

LOCAL_C_INCLUDES := $(INC) $(LOCAL_PATH)/include

LOCAL_CFLAGS := -mfloat-abi=softfp -mfpu=neon-vfpv4 -march=armv7-a
LOCAL_ARM_NEON := true
TARGET_ARCH_ABI := armeabi-v7a
LOCAL_ARM_MODE := arm

include $(BUILD_STATIC_LIBRARY)
