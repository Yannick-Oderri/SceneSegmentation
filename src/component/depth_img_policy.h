//
// Created by ynki9 on 3/5/20.
//

#ifndef PROJECT_EDGE_DEPTH_IMG_POLICY_H
#define PROJECT_EDGE_DEPTH_IMG_POLICY_H

#include "component/pipeline_policy.h"
#include "shader.hpp"
#include "res/resource_mgr.h"

struct EdgeParameters{
    int MorphologySize = 5;
};

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
    ContourAttributes* contour_attributes_;
    GLuint gl_depth_img_id_;
    GLFWwindow* parent_window_;
    GLFWwindow* current_window_;
    cv::Mat curve_disc_buffer_;


    /**
     * Process curve discontinuity
     * @param glContext
     */
    cv::Mat glProcessCurveDiscontinuity(GLFWwindow* const glContext, FrameElement* const frame_element, EdgeParameters* const eparams);

    /**
     * PRocess Dpeth Discontinuity
     * @param glContext
     * @param frame_element
     * @return
     */
    cv::Mat processDepthDiscontinuity(GLFWwindow *const glContext, FrameElement *const frame_element);

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
    bool executePolicy();

    /**
     * Frame data to be set for each frame to be processed
     * @param frame_element
     */
    void setFrameData(FrameElement* frame_element);

    /**
     * Return Contour attribtues generate during privious policy execution
     * @return  ContourAttribute or nullptr
     */
    ContourAttributes* getContourAttributes(){return this->contour_attributes_;};

    cv::Mat cuProcessCurveDiscontinuity(FrameElement *const frame_element, EdgeParameters *const edge_param);
};


#endif //PROJECT_EDGE_DEPTH_IMG_POLICY_H
