//
// Created by ynki9 on 3/5/20.
//

#include <glad/glad.h>
#include "depth_img_policy.h"
#include <boost/log/trivial.hpp>
#include <boost/timer/timer.hpp>
#include <utils/wiener_filter.h>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_glfw.h>

void updateBWParameters(Shader* const shdr_bw);

DepthImagePolicy::DepthImagePolicy(AppContext* const app_context):
framebuffer_width_(VIEWPORT_WIDTH),
framebuffer_height_(VIEWPORT_HEIGHT),
frame_element_(nullptr),
app_context_(app_context){

    // Initalize buffer element to store curve discontinuity
    unsigned char* curve_buffer = (unsigned char*)malloc(framebuffer_width_*framebuffer_height_*sizeof(int));
    this->curve_disc_buffer_ = cv::Mat(framebuffer_height_, framebuffer_width_, CV_8UC4, curve_buffer);
}


void updateEdgeParameters(EdgeParameters* const params);

cv::Mat processContourFrames(cv::Mat new_frame, std::deque<cv::Mat>& frame_queue){
    cv::Mat n_frame;
    new_frame.convertTo(n_frame, CV_16U);
    frame_queue.push_front(n_frame);
    if(frame_queue.size() > 16){
        frame_queue.pop_back();
    }
    const int k_size[] = {1, 2, 4, 6, 8, 10};
    cv::Mat t_frame1 = cv::Mat::zeros(new_frame.rows, new_frame.cols, n_frame.type());
    cv::Mat t_frame2 = cv::Mat::zeros(new_frame.rows, new_frame.cols, n_frame.type());

    for(int i=0; i < frame_queue.size(); i++){
        float k_size = pow((i+1)*0.15, 2)+1;
        cv::Size ks((int)k_size, (int)k_size);
        cv::blur(frame_queue[i], t_frame1, ks);
        t_frame2 = t_frame2 + t_frame1;
    }
    t_frame2 = t_frame2/frame_queue.size();

    return t_frame2;

}

bool DepthImagePolicy::executePolicy() {
    boost::timer::auto_cpu_timer profiler_image_policy("PROFILER: Edge Detect \t\t\t Time: %w secs\n");

    EdgeParameters edge_params;

    FrameElement* frame_element = this->frame_element_;
    if(frame_element == nullptr){
        BOOST_LOG_TRIVIAL(error) << "Null Frame element passed to Depth Image Policy";
        return false;
    }


    cv::Mat curve_disc = this->cuProcessCurveDiscontinuity(frame_element, &edge_params);
    cv::Mat depth_disc = this->processDepthDiscontinuity(this->parent_window_, frame_element);

    //curve_disc = cleanDiscontinuityOpr(curve_disc) * 255;
    //depth_disc = cleanDiscontinuityOpr(depth_disc) * 255;

    cv::Mat img = processContourFrames(curve_disc, contour_frame_queue_);// | depth_disc;
    cv::normalize(img, img, 255, 0, CV_MINMAX, CV_8U);
    cv::Mat img2;
    cv::threshold(img, img, 25, 255, CV_THRESH_BINARY);
    cv::imshow("Filtered Frames", img);
    //cv::imwrite("./results/filtered_contours.png", img);



//    cv::morphologyEx(depth_disc, depth_disc, cv::MORPH_CLOSE,
//                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6, 6)));
//    cv::morphologyEx(curve_disc, curve_disc, cv::MORPH_CLOSE,
//                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6, 6)));

//    cv::dilate(curve_disc, curve_disc,
//            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8)));

#ifdef OUTPUT_INTERNAL_STAGES
    int img_id = frame_element->getFrameID();
    string img_path = app_context_->getResMgr()->getOutDir() + "/" + to_string(img_id) + "_cd_dd"  + ".png";
    cv::imwrite(img_path, img);
    img_path = app_context_->getResMgr()->getOutDir() + "/" + to_string(img_id) + "_cd" + ".png";
    cv::imwrite(img_path, curve_disc);
#endif
#ifdef DEBUG_DEPTH_PLCY_IMG_RESULTS
    cv::imshow("dd & cd", img);
#endif

    cv::Mat skel = img.clone(); //(img.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;

    std::vector<std::vector<cv::Point> > t_contours;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours( skel, t_contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE, cv::Point(0, 0) );
    // ImageFrame *new_frame = new ImageFrame(frame->width, frame->height, 4*sizeof(float), new_buffer);
    // getOutQueue()->push(new_frame);


    /// Draw contours
    cv::Mat drawing = cv::Mat::zeros( skel.size(), CV_8UC3 );
    vector<int> to_delete;
    vector<int> to_add;
    for( int i = 0; i< t_contours.size(); i++ ) {
        double area = cv::contourArea(t_contours[i]);
        if(area < 350) continue;
        if(hierarchy[i][3] >= 0 && std::find(to_delete.begin(), to_delete.end(), hierarchy[i][3]) == to_delete.end())
            to_delete.push_back(hierarchy[i][3]);

        to_add.push_back(i);
        //contours.push_back(t_contours[i]);
        //cv::Scalar color = cv::Scalar(255,255, 255);
        //drawContours( drawing, t_contours, i, color, 0.8f, 8, hierarchy, 0, cv::Point() );
    }

    // Ranges must be sorted!
    std::sort(to_delete.begin(), to_delete.end());
    std::sort(to_add.begin(), to_add.end());

    std::vector<int> valid_contour; // Will contain the symmetric difference
    std::set_symmetric_difference(to_add.begin(), to_add.end(),
                                  to_delete.begin(), to_delete.end(),
                                  std::back_inserter(valid_contour));

    // second pass to delete enclosing parent contour
    for( int contour_index : valid_contour) {
        contours.push_back(t_contours[contour_index]);
        cv::Scalar color = cv::Scalar(255,255, 255);
        drawContours( drawing, t_contours, contour_index, color, 0.8f, 8, hierarchy, 0, cv::Point() );
        cv::imshow("contour_restults", drawing);
//        cv::waitKey(0);
    }

    /*if(contour_idx >= 0){
        cv::Scalar color = cv::Scalar( 255, 20, 20); //rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, contour_idx, color, 3, 8, hierarchy, 0, cv::Point() );

        std::vector<cv::Point> ransac_results = {performRansacOnCotour(t_image, contours, contour_idx, point, hierarchy)};
        std::vector<std::vector<cv::Point>> results;
        results.push_back(ransac_results);
        cv::fillPoly(drawing, results, cv::Scalar(32, 32, 240));
    }*/
#ifdef DEBUG_DEPTH_PLCY_IMG_RESULTS
    /// Show in a window
//        cv::namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    cv::imshow( "Contours", drawing );
//    cv::imwrite("/home/ynki9/Desktop/ynk_img/test11/contour.png", drawing);
    //    cv::imwrite("/home/ynki9/Desktop/ynk_img/test11/depth_disc.png", depth_disc);
    //    cv::imwrite("/home/ynki9/Desktop/ynk_img/test11/curve_disc.png", curve_disc);
    cv::waitKey(0);
#endif

#ifdef OUTPUT_INTERNAL_STAGES
    img_id = frame_element->getFrameID();
    img_path = app_context_->getResMgr()->getOutDir() + "/" + to_string(img_id) + "contours"  + ".png";
    cv::imwrite(img_path, drawing);
#endif

    /// Add edge information to frame element
    //frame_element->setEdgeData(depth_disc, curve_disc, drawing);

    /// Push Contour data along with frame data to next stage in pipeline
    //this->contour_attributes_ = new ContourAttributes(frame_element, contours);
    return true;
}

void DepthImagePolicy::intialize() {
    this->intializeGLParams(this->app_context_);

    // Initialize Dear ImGui
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
// Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(this->current_window_, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
// Setup Dear ImGui style
    ImGui::StyleColorsDark();
}

void DepthImagePolicy::intializeGLParams(AppContext* const ctxt) {
    // glfw: intialize
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint( GLFW_DOUBLEBUFFER, GL_TRUE ); // use single buffer to avoid vsync

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    /// Enable Offscreen rendering
    //glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

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
    this->shdr_median_blur_.enableFramebuffer(true);
    //this->shdr_sobel_.enableFramebuffer(true);
    this->shdr_blk_whte_.enableFramebuffer(true);


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

    this->shdr_blk_whte_.use();
    this->shdr_blk_whte_.setInt("iChannel0", 0);

    this->shdr_bilateral_.use();
    this->shdr_bilateral_.setVec2("iResolution", glm::vec2(framebuffer_width_, framebuffer_height_));
    this->shdr_bilateral_.setInt("iChannel0", 0);

    this->shdr_median_blur_.use();
    this->shdr_median_blur_.setVec2("Tinvsize", glm::vec2(1.0f, 1.0f));
    this->shdr_median_blur_.setInt("iChannel0", 0);

    this->shdr_sobel_.use();
    y_sobel = glm::mat3x3(glm::vec3(-1,-1,-1),
            glm::vec3(-1, 8,-1), glm::vec3(-1,-1,-1));
    this->shdr_sobel_.setMat3("convolutionMatrix_y", y_sobel);
    x_sobel = glm::mat3x3(glm::vec3(-1,-1,-1),
            glm::vec3(-1, 8,-1), glm::vec3(-1,-1,-1));
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
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_DEPTH_COMPONENT, framebuffer_width_,
            framebuffer_height_, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

    double previousTime = glfwGetTime();
    int frameCount = 0;
}


cv::Mat DepthImagePolicy::cuProcessCurveDiscontinuity(FrameElement* const frame_element, EdgeParameters* const edge_param){
    boost::timer::auto_cpu_timer profiler_process_CD("PROFILER: Curve Disc Opr \t\t\t Time: %w secs\n");
    cv::Mat depth_img = frame_element->getDepthFrameData()->getDepthImage();
    cv::Mat res = cuCurveDiscOperation(depth_img);
    return res;
}

cv::Mat DepthImagePolicy::glProcessCurveDiscontinuity(GLFWwindow* const glContext,
        FrameElement* const frame_element, EdgeParameters* const edge_params) {
    glfwMakeContextCurrent(this->current_window_);
    glfwPollEvents();

    this->shdr_normal_.use();

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    /// Duplicate for curve discontinuity
    cv::Mat normalized_dd = frame_element->getDepthFrameData()->getNDepthImage();
    double min_val, max_val;
    cv::minMaxLoc(normalized_dd, &min_val, &max_val);

    // feed inputs to dear imgui, start new frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    /// Calculate Curve Discontinuity
    glBindTexture(GL_TEXTURE_RECTANGLE, this->gl_depth_img_id_);
    glTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, framebuffer_width_, framebuffer_height_,
            GL_DEPTH_COMPONENT, GL_FLOAT, normalized_dd.ptr());
    //glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB, framebuffer_width_, framebuffer_height_, 0, GL_RGB, GL_UNSIGNED_BYTE, image.ptr());
    //glGenerateMipmap(GL_TEXTURE_RECTANGLE);

/// Rendering Routine
    // first pass -- Gradient Shader
    this->shdr_normal_.use();
    glBindFramebuffer(GL_FRAMEBUFFER, this->shdr_median_blur_.getFramebufferTextureID());
    glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // render viewport rectangle
    glActiveTexture(GL_TEXTURE0); // TEXTURE0 image location
    glBindTexture(GL_TEXTURE_RECTANGLE, this->gl_depth_img_id_);
    this->shdr_normal_.use();
    glBindVertexArray(this->fb_quad_vao_);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    // Perform Media Blur
    this->shdr_median_blur_.use();
    glBindFramebuffer(GL_FRAMEBUFFER, this->shdr_median_blur_.getFramebufferTextureID());
    glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindVertexArray(this->fb_quad_vao_);
    glActiveTexture(GL_TEXTURE0);
    glad_glBindTexture(GL_TEXTURE_RECTANGLE, this->shdr_normal_.getFramebufferTextureID());
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    // second pass -- black and white
    this->shdr_blk_whte_.use();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindVertexArray(this->fb_quad_vao_);
    glActiveTexture(GL_TEXTURE0); // TEXTURE0 image location
    glBindTexture(GL_TEXTURE_RECTANGLE, this->shdr_median_blur_.getFramebufferTextureID());
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    // render your GUI
    updateBWParameters(&this->shdr_blk_whte_);
    updateEdgeParameters(edge_params);

        // Render dear imgui into screen
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    int display_w, display_h;
    glfwGetFramebufferSize(this->current_window_, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glfwSwapBuffers(this->current_window_);
    glFlush();

    glReadPixels(0, 0, framebuffer_width_, framebuffer_height_, GL_RGBA,
            GL_UNSIGNED_BYTE, this->curve_disc_buffer_.ptr());

    std::vector<cv::Mat> channels(4);
    cv::split(this->curve_disc_buffer_, channels);

    cv::Mat res;

    WienerFilter(channels[0], channels[1], -1, cv::Size(9, 9));

    cv::imshow("Wiener Result", channels[1]);

    cv::medianBlur(channels[0], channels[1], 5);
    cv::Canny(channels[1], res, 200.0, 240.0);
    cv::imshow("Wiener 2", res);

    cv::waitKey(0);
    return res;
}


cv::Mat DepthImagePolicy::processDepthDiscontinuity(GLFWwindow* const glContext,
        FrameElement* const frame_element){
    // Calculate Depth Discontinuity
    cv::Mat t_dimg = frame_element->getDepthFrameData()->getNDepthImage();
    cv::Mat depth_disc;
    t_dimg.convertTo(depth_disc, CV_8U, 255, 0);


    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    cv::Mat depth_dst;

    clahe->apply(depth_disc, depth_dst);
    cv::Canny(depth_dst, depth_disc, 43.0, 90.0);

    return depth_disc;
}

void updateBWParameters(Shader* const shdr_bw){
    float val = 5.0;
    float min = -10.0;
    float max = 10.0;
    static float bnw_coeffs[6] = {-val, -val, val, val, -val, -val};
    ImGui::Begin("Black and White");
    ImGui::SliderFloat3("RYG", bnw_coeffs, min, max);
    ImGui::SliderFloat3("CBM", (bnw_coeffs + 3), min, max);
    ImGui::End();

    shdr_bw->setFloatv("coeff_values", bnw_coeffs, 6);
}

void updateEdgeParameters(EdgeParameters* const params){
    static EdgeParameters tparams;
    ImGui::Begin("Edge Params");
    ImGui::SliderInt("Element Size", &tparams.MorphologySize, 1, 12);
    ImGui::End();

    params->MorphologySize = tparams.MorphologySize;
}


void DepthImagePolicy::setFrameData(FrameElement *frame_element) {
    this->frame_element_ = frame_element;
}