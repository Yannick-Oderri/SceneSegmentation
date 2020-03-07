//
// Created by ynki9 on 3/5/20.
//

#ifndef PROJECT_EDGE_DEPTH_IMG_POLICY_H
#define PROJECT_EDGE_DEPTH_IMG_POLICY_H

#include "component/pipeline_policy.h"
#include "shader.hpp"
#include "res/resource_mgr.h"

class DepthImagePolicy: PipelinePolicy {
private:
    int framebuffer_width_;
    int framebuffer_height_;
    Shader shdr_normal_;
    Shader shdr_median_blur_;
    Shader shdr_sobel_;
    Shader shdr_bilateral_;
    Shader shdr_blk_whte_;
    unsigned int fb_quad_vbo_, fb_quad_vao_, fb_quad_ebo_;
    AppContext* app_context_;
    FrameElement* frame_element_;
    GLuint gl_depth_img_id_;
    GLFWwindow* parent_window_;
    GLFWwindow* current_window_;


    /**
     * Process curve discontinuity
     * @param glContext
     */
    void glProcessCurveDiscontinuity(GLFWwindow* const glContext, FrameElement* const frame_element);

    /**
     * PRocess depth Discontinuity
     */
    void processDepthDiscontinuity();

    /**
     * Intialize GL parameters for policy execution
     * @param ctxt
     */
    void intializeGLParams(AppContext* const ctxt);

public:
    /**
     * Policy Constructor
     */
    DepthImagePolicy(AppContext* const app_context);

    /**
     * Policy Initialization routine
     * @param parentContext
     */
    void intialize();

    /**
     * Policy execution routine
     */
    void executePolicy();

    /**
     * Frame data to be set for each frame to be processed
     * @param frame_element
     */
    void setFrameData(FrameElement* frame_element);
};


#endif //PROJECT_EDGE_DEPTH_IMG_POLICY_H
