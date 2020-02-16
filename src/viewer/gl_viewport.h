//
// Created by ynk on 1/17/20.
//

#ifndef PROJECT_EDGE_GL_VIEWPORT_H
#define PROJECT_EDGE_GL_VIEWPORT_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <opencv2/opencv.hpp>

#include "res/resource_mgr.h"
#include "gl_camera.h"
#include "shader.hpp"


class GLViewport
{
public:
    /**
     * Pointcloud Viewport Constructor
     * @param app_ctx
     */
    GLViewport(AppContext& app_ctx);

    /**
     * Descructor
     */
    ~GLViewport(void);

    /**
     * Renders current image frame
     * @param camera_image
     * @param point_cloud
     * @return
     */
    int RenderFrame(cv::Mat camera_image,cv::Mat point_cloud);

    /**
     * Terminate rendering
     */
    void terminate();

    /**
     * GLFW should close
     * @return
     */
    bool ShouldClose() const {return glfwWindowShouldClose(this->app_context_->getGLContext());};

private:
    int initialize(AppContext&);
    int CreateGeometries();
    void on_scroll(GLFWwindow* window, double xoffset, double yoffset);
    void on_mouse(GLFWwindow* window, double xpos, double ypos);
    void on_window_resize(GLFWwindow* window, int width, int height);
    void ProcessInput();
    void ShowStereoCamera(bool show) { show_stereo_camera_=show;};

    /// Private Fields

    std::unique_ptr<GLCamera> camera_;
    Shader cam_render_shader_;
    unsigned int cam_render_texture_;
    Shader point_cloud_shader_;
    cv::Mat image;
    const GLuint window_width_;
    const GLuint window_height_;
    AppContext* const app_context_;
    bool first_mouse_;
    float last_x_;
    float last_y_;
    unsigned int camera_quad_vbo_, camera_quad_vao_, camera_quad_ebo_;
    unsigned int point_cloud_vao_,point_cloud_vbo_;
    float delta_time_;
    float last_frame_time_;
    bool show_stereo_camera_;
    GLFWwindow* window_;
};


#endif //PROJECT_EDGE_GL_VIEWPORT_H
