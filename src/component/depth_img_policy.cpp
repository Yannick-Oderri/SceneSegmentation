//
// Created by ynki9 on 3/5/20.
//

#include <glad/glad.h>
#include "depth_img_policy.h"
#include <boost/log/trivial.hpp>


DepthImagePolicy::DepthImagePolicy(AppContext* const app_context):
framebuffer_width_(VIEWPORT_WIDTH),
framebuffer_height_(VIEWPORT_HEIGHT),
frame_element_(nullptr),
app_context_(app_context){}


void DepthImagePolicy::executePolicy() {
    FrameElement* const frame_element = this->frame_element_;
    if(frame_element == nullptr){
        BOOST_LOG_TRIVIAL(error) << "Null Frame element passed to Depth Image Policy";
        return;
    }



    this->glProcessCurveDiscontinuity(this->parent_window_, frame_element);
}

void DepthImagePolicy::intialize() {
    this->intializeGLParams(this->app_context_);
}

void DepthImagePolicy::intializeGLParams(AppContext* const ctxt) {
    // glfw: intialize
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint( GLFW_DOUBLEBUFFER, GL_FALSE ); // use single buffer to avoid vsync

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    /// Enable Offscreen rendering
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    this->current_window_ = glfwCreateWindow(framebuffer_width_, framebuffer_height_, "Curve Discontinuity Framebuffer", nullptr, ctxt->getGLContext());
    if (this->current_window_ == NULL){
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(this->current_window_);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return;
    }
    GLuint err;

    this->shdr_normal_ = ctxt->getResMgr()->loadShader("shader_1.vs", "depth_normals.fs");
    this->shdr_median_blur_ = ctxt->getResMgr()->loadShader("shader_1.vs", "median_blur.fs");
    this->shdr_sobel_ = ctxt->getResMgr()->loadShader("shader_1.vs", "sobel.fs");
    this->shdr_bilateral_ = ctxt->getResMgr()->loadShader("shader_1.vs", "bilateral_blur.fs");
    this->shdr_blk_whte_ = ctxt->getResMgr()->loadShader("shader_1.vs", "black_n_white.fs");


    // Set all shaders to
    this->shdr_normal_.enableFramebuffer(true);
    //this->shdr_bilateral_.enableFramebuffer(true);
    //this->shdr_median_blur_.enableFramebuffer(true);
    //this->shdr_sobel_.enableFramebuffer(true);
    //this->shdr_blk_whte_.enableFramebuffer(true);


    err = glGetError();
    if(err != GL_NO_ERROR){
        std::cout << "Image failed to load "<< err << std::endl;
        return;
    }

    // depth image shader reference
    GLuint dimg_location = glGetUniformLocation(this->shdr_normal_.ID, "dmap");
    err = glGetError();
    if(err != GL_NO_ERROR){
        std::cout << "Get Uniform failed. "<< err << std::endl;
        return;
    }

    // Frame reference location
    float vertices[] = {
            // positions         // colors
            1.0f, -1.0f, 0.0f,  1.0f, 0.0f, 0.0f, VIEWPORT_WIDTH, 0.0f,  // bottom right
            -1.0f, -1.0f, 0.0f,  0.0f, 1.0f, 0.0f, 0.0f, 0.0f, // bottom left
            -1.0f,  1.0f, 0.0f,  0.0f, 0.0f, 1.0f, 0.0f, VIEWPORT_HEIGHT,// top left
            1.0f,  1.0f, 0.0f,  0.0f, 0.0f, 1.0f, VIEWPORT_WIDTH, VIEWPORT_HEIGHT,// top  right
    };

    unsigned int indices[] = {
            0, 1, 2,
            2, 3, 0
    };

    glGenVertexArrays(1, &this->fb_quad_vao_);
    glGenBuffers(1, &this->fb_quad_vbo_);
    glGenBuffers(1, &this->fb_quad_ebo_);

    // bind the Vertex Array object first, then bind and set vertex buffers,  then configure vertex attributes
    glBindVertexArray(this->fb_quad_vao_);

    glBindBuffer(GL_ARRAY_BUFFER, this->fb_quad_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->fb_quad_ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    //glBindVertexArray(0);

    this->shdr_normal_.use();
    // glUniform1i(glGetUniformLocation(shader.ID, "dmap"), 0);
    glUniform1i(dimg_location, 0);
    err = glGetError();
    if(err != GL_NO_ERROR){

        std::cout << "Set 1Uniform image failed "<< err << " "<< dimg_location << std::endl;
        return;
    }

    glm::mat3x3 y_sobel(glm::vec3(1,2,1), glm::vec3(0, 0, 0), glm::vec3(-1,-2,-1));
    this->shdr_normal_.setMat3("convolutionMatrix_y", y_sobel);
    glm::mat3x3 x_sobel(glm::vec3(1,0,-1), glm::vec3(2, 0, -2), glm::vec3(1,0,-1));
    this->shdr_normal_.setMat3("convolutionMatrix_x", x_sobel);
    this->shdr_normal_.setInt("dmap", 0);

    this->shdr_bilateral_.use();
    this->shdr_bilateral_.setVec2("iResolution", glm::vec2(framebuffer_width_, framebuffer_height_));
    this->shdr_bilateral_.setInt("iChannel0", 0);

    this->shdr_median_blur_.use();
    this->shdr_median_blur_.setVec2("Tinvsize", glm::vec2(1.0f, 1.0f));
    this->shdr_median_blur_.setInt("iChannel0", 0);

    this->shdr_sobel_.use();
    y_sobel = glm::mat3x3(glm::vec3(-1,-1,-1), glm::vec3(-1, 8,-1), glm::vec3(-1,-1,-1));
    this->shdr_sobel_.setMat3("convolutionMatrix_y", y_sobel);
    x_sobel = glm::mat3x3(glm::vec3(-1,-1,-1), glm::vec3(-1, 8,-1), glm::vec3(-1,-1,-1));
    this->shdr_sobel_.setMat3("convolutionMatrix_x", x_sobel);
    this->shdr_sobel_.setInt("dmap", 0);

// Setup input image gl texture location
    GLuint textID;
    glGenTextures(1, &this->gl_depth_img_id_);
    //std::cout << glGetError() << std::endl; // returns 0 (no error)
    glBindTexture(GL_TEXTURE_RECTANGLE, this->gl_depth_img_id_);
    // std::cout << glGetError() << std::endl; // returns 0 (no error)
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // generate an empty gl texture location
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_DEPTH_COMPONENT, framebuffer_width_, framebuffer_height_, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

    double previousTime = glfwGetTime();
    int frameCount = 0;
}


void DepthImagePolicy::glProcessCurveDiscontinuity(GLFWwindow* const glContext, FrameElement* const frame_element) {
    glfwMakeContextCurrent(this->current_window_);

    this->shdr_normal_.use();

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    /// Duplicate for curve discontinuity
    cv::Mat normalized_dd = frame_element->getDepthFrameData()->getNDepthImage();
    double min_val, max_val;
    cv::minMaxLoc(normalized_dd, &min_val, &max_val);

    /// Calculate Curve Discontinuity
    glBindTexture(GL_TEXTURE_RECTANGLE, this->gl_depth_img_id_);
    glTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, framebuffer_width_, framebuffer_height_, GL_DEPTH_COMPONENT, GL_FLOAT, normalized_dd.ptr());
    //glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB, framebuffer_width_, framebuffer_height_, 0, GL_RGB, GL_UNSIGNED_BYTE, image.ptr());
    //glGenerateMipmap(GL_TEXTURE_RECTANGLE);

/// Rendering Routine
    // first pass -- Gradient Shader
    this->shdr_normal_.use();
    glBindFramebuffer(GL_FRAMEBUFFER, this->shdr_normal_.getFramebufferTextureID());
    glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // render viewport rectangle
    glActiveTexture(GL_TEXTURE0); // TEXTURE0 image location
    glBindTexture(GL_TEXTURE_RECTANGLE, this->gl_depth_img_id_);
    this->shdr_normal_.use();
    glBindVertexArray(this->fb_quad_vao_);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    // second pass -- black and white
//    glBindFramebuffer(GL_FRAMEBUFFER, blur_shader_1.getFramebuferID());
//    glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
//    glClear(GL_COLOR_BUFFER_BIT);
//
//    blur_shader_1.use();
//    glBindVertexArray(VAO);
//    glActiveTexture(GL_TEXTURE0); // TEXTURE0 image location
//    glBindTexture(GL_TEXTURE_RECTANGLE, shader.getFramebufferTextureID());
//    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    // // third pass -- bilateral filter shader
    // glBindFramebuffer(GL_FRAMEBUFFER, blur_shader_2.getFramebuferID());
    // glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
    // glClear(GL_COLOR_BUFFER_BIT);

    // blur_shader_2.use();
    // glBindVertexArray(VAO);
    // glActiveTexture(GL_TEXTURE0); // TEXTURE0 image location
    // glBindTexture(GL_TEXTURE_RECTANGLE, blur_shader_1.getFramebufferTextureID());
    // glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    // fourth pass -- bilateral filter shader
//    glBindFramebuffer(GL_FRAMEBUFFER, this->shdr_bilateral_.getFramebuferID());
//    glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
//    glClear(GL_COLOR_BUFFER_BIT);
//
//    this->shdr_bilateral_.use();
//    glBindVertexArray(this->fb_quad_vao_);
//    glActiveTexture(GL_TEXTURE0); // TEXTURE0 image location
//    glBindTexture(GL_TEXTURE_RECTANGLE, this->shdr_median_blur_.getFramebufferTextureID());
//    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
//
//    // // // fifth pass -- bilateral filter shader
//    glBindFramebuffer(GL_FRAMEBUFFER, this->shdr_sobel_.getFramebuferID());
//    glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
//    glClear(GL_COLOR_BUFFER_BIT);
//
//    this->shdr_sobel_.use();
//    glBindVertexArray(this->fb_quad_vao_);
//    glActiveTexture(GL_TEXTURE0); // TEXTURE0 image location
//    glBindTexture(GL_TEXTURE_RECTANGLE, this->shdr_median_blur_.getFramebufferTextureID());
//    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    //glfwSwapBuffers(current_window_);
    glFlush();
    glfwPollEvents();



    unsigned char* new_buffer = (unsigned char*)malloc(framebuffer_width_*framebuffer_height_*sizeof(int));
    glReadPixels(0, 0, framebuffer_width_, framebuffer_height_, GL_RGBA, GL_UNSIGNED_BYTE, new_buffer);
    cv::Mat mat = cv::Mat(framebuffer_height_, framebuffer_width_, CV_8UC4, new_buffer);


    cv::imshow("Curve Disc", mat);
    cv::waitKey(0);

}



void DepthImagePolicy::setFrameData(FrameElement *frame_element) {
    this->frame_element_ = frame_element;
}