//
// Created by ynki9 on 12/31/19.
//

#ifndef PROJECT_EDGE_GL_EDGE_DISC_FILTER_H
#define PROJECT_EDGE_GL_EDGE_DISC_FILTER_H

#include <libfreenect2/libfreenect2.hpp>
#include "dataflow/pipeline_filter.h"
#include "frame.h"

using namespace std;


/**
 * Contour Extractor pipeline filter
 */
class GLEdgeDiscFilter:  public PipeFilter<FrameElement*, ContourAttributes*>{
private:
    int viewport_width_;
    int viewport_height_;
    GLFWwindow* parent_window_;

public:
    /**
     * Construct GLEdgeDisc
     * @param in_queue Input from previous pipeline Pipe
     */
    GLEdgeDiscFilter(QueueClient<FrameElement*>* in_queue):
    PipeFilter(in_queue, new QueueClient<ContourAttributes* >()),
    viewport_width_(VIEWPORT_WIDTH),
    viewport_height_(VIEWPORT_HEIGHT),
    parent_window_(nullptr){}

    /**
     * Initialize Pipeline filter
     */
    void initialize();

    /**
     * Execute Pipeline filter thread. Functoin loops until application ends
     */
    void start();

    /**
     * Sets GLFW parent context/window
     * @param parent_window
     */
    void setParentContext(GLFWwindow *parent_window);

};


#endif //PROJECT_EDGE_GL_EDGE_DISC_FILTER_H
