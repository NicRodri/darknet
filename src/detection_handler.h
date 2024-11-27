#ifndef DETECTION_HANDLER_H
#define DETECTION_HANDLER_H

#include "/root/darknet/include/darknet.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single wrapper function
void process_frame(void* show_img, detection* dets, int nboxes, float thresh, char** names, int classes);

#ifdef __cplusplus
}
#endif

#endif // DETECTION_HANDLER_H
