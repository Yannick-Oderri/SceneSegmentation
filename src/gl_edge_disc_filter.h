//
// Created by ynki9 on 12/31/19.
//

#ifndef PROJECT_EDGE_GL_EDGE_DISC_FILTER_H
#define PROJECT_EDGE_GL_EDGE_DISC_FILTER_H

#include <libfreenect2/libfreenect2.hpp>
#include "dataflow/pipeline_filter.h"


class GLEdgeDiscFilter:  public PipeFilter<FrameRegistration*, ImageFrame*>{
private:
    int viewport_width_;
    int viewport_height_;

public:
    GLEdgeDiscFilter(QueueClient<ImageFrame*>* in_queue):
    PipeFilter(in_queue, new QueueClient<ImageFrame*>()),
    viewport_width_(VIEWPORT_WIDTH),
    viewport_height_(VIEWPORT_HEIGHT){}

    void initialize();
    void start();
};


#endif //PROJECT_EDGE_GL_EDGE_DISC_FILTER_H
