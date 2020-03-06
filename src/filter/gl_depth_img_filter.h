//
// Created by ynki9 on 3/6/20.
//

#ifndef PROJECT_EDGE_GL_DEPTH_IMG_FILTER_H
#define PROJECT_EDGE_GL_DEPTH_IMG_FILTER_H

#include <libfreenect2/libfreenect2.hpp>
#include "dataflow/pipeline_filter.h"
#include "frame.h"
#include "component/depth_img_policy.h"

using namespace std;


/**
 * Contour Extractor pipeline filter
 */
class GLDepthImageFilter:  public PipeFilter<FrameElement*, ContourAttributes*>{
private:
    int viewport_width_;
    int viewport_height_;
    DepthImagePolicy* const exec_policy;


public:
    /**
     * Construct GLDepthImageProcessor
     * @param in_queue Input from previous pipeline Pipe
     */
    GLDepthImageFilter(QueueClient<FrameElement*>* in_queue, DepthImagePolicy* const processing_policy):
    PipeFilter(in_queue, new QueueClient<ContourAttributes* >()),
    viewport_width_(VIEWPORT_WIDTH),
    viewport_height_(VIEWPORT_HEIGHT),
    exec_policy(processing_policy){}

    /**
     * Initialize Pipeline filter
     */
    void initialize();

    /**
     * Execute Pipeline filter thread. Functoin loops until application ends
     */
    void start();
};
#endif //PROJECT_EDGE_GL_DEPTH_IMG_FILTER_H
