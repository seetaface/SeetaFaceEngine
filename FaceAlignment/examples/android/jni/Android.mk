LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE_TAGS := optional
LOCAL_MODULE := FaceAlignment

DIRS = FaceAlignment/src \
	FaceAlignment/src/classifier \
	FaceAlignment/src/feat \
	FaceAlignment/src/io \
	FaceAlignment/src/util \
	FaceAlignment/include

INC  = $(foreach dir,$(DIRS),$(LOCAL_PATH)/$(dir))
SRC  = $(foreach dir,$(DIRS),$(wildcard $(LOCAL_PATH)/$(dir)/*.c))
SRC += $(foreach dir,$(DIRS),$(wildcard $(LOCAL_PATH)/$(dir)/*.cpp))

LOCAL_SRC_FILES := $(SRC:$(LOCAL_PATH)/%=%)

LOCAL_C_INCLUDES := $(INC)

LOCAL_CFLAGS := -Wall

include $(BUILD_STATIC_LIBRARY)
