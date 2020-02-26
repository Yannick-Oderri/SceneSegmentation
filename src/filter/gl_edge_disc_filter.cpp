//
// Created by ynki9 on 12/31/19.
//

#define GLFW_INCLUDE_NONE


#include <execinfo.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <boost/log/trivial.hpp>
#include "shader.hpp"

#include "gl_edge_disc_filter.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include "cuda_ransac_kernel.h"
#endif


int findEnclosingContour(std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Vec4i> &hierarchy, cv::Point2d &point, int idx=-1);
#ifdef WITH_CUDA
void performRansacOnCotour(cv::Mat &img, std::vector<std::vector<cv::Point>> &contours, int idx, cv::Point2d &, std::vector<cv::Vec4i> &);
#endif

void handler(int sig) {
    void *array[20];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 20);

    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height){
    // Ensures frabebuffer size matches windows size
    if (window != nullptr){

        glViewport(0, 0, width, height);
    }
}

void GLEdgeDiscFilter::initialize() {

}

void GLEdgeDiscFilter::setParentContext(GLFWwindow* parent_window){
    this->parent_window_ = parent_window;
}

void GLEdgeDiscFilter::start() {
    // signal(SIGABRT, handler);   // install our handler

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
    // glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(viewport_width_, viewport_height_, "Shader", nullptr, this->parent_window_);
    if (window == NULL){
        std::cout << "Faineld to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return;
    }
    GLuint err;

    Shader blur_shader_1("../data/shaders/shader_1.vs", "../data/shaders/bilateral_blur.fs");
    Shader median_blur_shader("../data/shaders/shader_1.vs", "../data/shaders/median_blur.fs");
    Shader sobel_shader("../data/shaders/shader_1.vs", "../data/shaders/sobel.fs");
    Shader shader("../data/shaders/shader_1.vs", "../data/shaders/shader_1.fs");
    shader.enableFramebuffer(true);
    blur_shader_1.enableFramebuffer(true);
    //blur_shader_2.enableFramebuffer(true);
    median_blur_shader.enableFramebuffer(true);

    err = glGetError();
    if(err != GL_NO_ERROR){
        std::cout << "Image failed to load "<< err << std::endl;
        return;
    }

    GLuint dimg_location = glGetUniformLocation(shader.ID, "dmap");
    err = glGetError();
    if(err != GL_NO_ERROR){
        std::cout << "Get Uniform failed. "<< err << std::endl;
        return;
    }

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

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // bind the Vertex Array object first, then bind and set vertex buffers,  then configure vertex attributes
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
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
    // glBindVertexArray(0);

    shader.use();
    // glUniform1i(glGetUniformLocation(shader.ID, "dmap"), 0);
    glUniform1i(dimg_location, 0);
    err = glGetError();
    if(err != GL_NO_ERROR){

        std::cout << "Set 1Uniform image failed "<< err << " "<< dimg_location << std::endl;
        return;
    }

    glm::mat3x3 y_sobel(glm::vec3(1,2,1), glm::vec3(0, 0, 0), glm::vec3(-1,-2,-1));
    shader.setMat3("convolutionMatrix_y", y_sobel);
    glm::mat3x3 x_sobel(glm::vec3(1,0,-1), glm::vec3(2, 0, -2), glm::vec3(1,0,-1));
    shader.setMat3("convolutionMatrix_x", x_sobel);
    shader.setInt("dmap", 0);

    blur_shader_1.use();
    blur_shader_1.setVec2("iResolution", glm::vec2(viewport_width_, viewport_height_));
    blur_shader_1.setInt("iChannel0", 0);

    median_blur_shader.use();
    median_blur_shader.setVec2("Tinvsize", glm::vec2(1.0f, 1.0f));
    median_blur_shader.setInt("iChannel0", 0);

    sobel_shader.use();
    y_sobel = glm::mat3x3(glm::vec3(-1,-1,-1), glm::vec3(-1, 8,-1), glm::vec3(-1,-1,-1));
    sobel_shader.setMat3("convolutionMatrix_y", y_sobel);
    x_sobel = glm::mat3x3(glm::vec3(-1,-1,-1), glm::vec3(-1, 8,-1), glm::vec3(-1,-1,-1));
    sobel_shader.setMat3("convolutionMatrix_x", x_sobel);
    sobel_shader.setInt("dmap", 0);

    double previousTime = glfwGetTime();
    int frameCount = 0;

    // transfer texture to  texture
//    cv::startWindowThread();
    // render loop
    // -----------
    shader.use();
    while (!glfwWindowShouldClose(window))
    {
        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        getInQueue()->waitData();
        FrameElement* frame_element = getInQueue()->front();

        /// Duplicate frame for depth discontinuity
        unsigned char* dd_buffer = (unsigned char*)malloc(viewport_width_*viewport_height_*sizeof(float));
        memcpy(dd_buffer, frame_element->getDepthFrameData()->getData(), viewport_width_*viewport_height_*sizeof(float));
        libfreenect2::Frame dd_frame(viewport_width_, viewport_height_, sizeof(float), dd_buffer);

        /// Duplicate for curve discontinuity
        cv::Mat curve_discontinuity;
        cv::Mat t_curve_discontinuity(frame_element->getDepthFrameData()->getHeight(), frame_element->getDepthFrameData()->getWidth(), CV_32F, (unsigned char*) frame_element->getDepthFrameData()->getData());
        double min_val, max_val;
        cv::minMaxLoc(t_curve_discontinuity, &min_val, &max_val);
        // cv::flip(t_curve_discontinuity, t_curve_discontinuity, 0);
        t_curve_discontinuity.convertTo(curve_discontinuity, CV_32F, 1.0/max_val, 0);

        processInput(window);

        // BOOST_LOG_TRIVIAL(info) << "Receiving edge discription frame:" << frameCount;
/// Calcuate curve Discontinoutiyu
        GLuint textID;
        glGenTextures(1, &textID);
        //std::cout << glGetError() << std::endl; // returns 0 (no error)
        glBindTexture(GL_TEXTURE_RECTANGLE, textID);
        // std::cout << glGetError() << std::endl; // returns 0 (no error)
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_DEPTH_COMPONENT, viewport_width_, viewport_height_, 0, GL_DEPTH_COMPONENT, GL_FLOAT, curve_discontinuity.ptr());
        //glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB, viewport_width_, viewport_height_, 0, GL_RGB, GL_UNSIGNED_BYTE, image.ptr());
        //glGenerateMipmap(GL_TEXTURE_RECTANGLE);

        // profile time
        // -----
        double currentTime = glfwGetTime();
        frameCount++;
        BOOST_LOG_TRIVIAL(info) << "PROFILER: " << currentTime-previousTime;
        previousTime = currentTime;
        if (currentTime - previousTime >= 1){
            std::cout << "Frame Rate: " << frameCount << std::endl;
//            frameCount = 0;
        }
        // input
        // -----


        // render
        // first pass -- Gradient Shader
        shader.use();
        glBindFramebuffer(GL_FRAMEBUFFER, shader.getFramebuferID());
        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // render viewport rectangle
        glActiveTexture(GL_TEXTURE0); // TEXTURE0 image location
        glBindTexture(GL_TEXTURE_RECTANGLE, textID);
        shader.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        // second pass -- bilateral filter shader
        glBindFramebuffer(GL_FRAMEBUFFER, blur_shader_1.getFramebuferID());
        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        blur_shader_1.use();
        glBindVertexArray(VAO);
        glActiveTexture(GL_TEXTURE0); // TEXTURE0 image location
        glBindTexture(GL_TEXTURE_RECTANGLE, shader.getFramebufferTextureID());
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

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
        glBindFramebuffer(GL_FRAMEBUFFER, median_blur_shader.getFramebuferID());
        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        median_blur_shader.use();
        glBindVertexArray(VAO);
        glActiveTexture(GL_TEXTURE0); // TEXTURE0 image location
        glBindTexture(GL_TEXTURE_RECTANGLE, blur_shader_1.getFramebufferTextureID());
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        // // // fifth pass -- bilateral filter shader
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        sobel_shader.use();
        glBindVertexArray(VAO);
        glActiveTexture(GL_TEXTURE0); // TEXTURE0 image location
        glBindTexture(GL_TEXTURE_RECTANGLE, median_blur_shader.getFramebufferTextureID());
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);



// Calculate Depth Discontinuity

        cv::Mat t_image(viewport_height_, viewport_width_, CV_32F, dd_frame.data);
        min_val, max_val;
        cv::minMaxLoc(t_image, &min_val, &max_val);
        t_image.convertTo(t_image, CV_32F, 1.0/max_val, 0);
        cv::Mat depth_disc;
        t_image.convertTo(depth_disc, CV_8U, 255, 0);


        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(4);
        cv::Mat depth_dst;

        clahe->apply(depth_disc, depth_dst);
        cv::Mat depth_canny;
        cv::Canny(depth_dst, depth_canny, 43.0, 90.0);


//        glfwSwapBuffers(window);
        glFlush();
        glfwPollEvents();

//        cv::flip(depth_canny, depth_canny, 0);
//        cv::flip(depth_canny, depth_canny, 1);
//        cv::imshow("Depth Edges", depth_canny);
//        cv::waitKey(0);


        unsigned char* new_buffer = (unsigned char*)malloc(dd_frame.width*dd_frame.height*sizeof(int));
        glReadPixels(0, 0, dd_frame.width, dd_frame.height, GL_RGBA, GL_UNSIGNED_BYTE, new_buffer);
        cv::Mat mat = cv::Mat(viewport_height_, viewport_width_, CV_8UC4, new_buffer);

        std::vector<cv::Mat> channels(4);
        cv::split(mat, channels);
        t_image.convertTo(depth_disc, CV_8U, 255, 0);
        cv::threshold(channels[0], channels[0], 10, 255, cv::THRESH_BINARY);
        cv::Mat tt_mat = channels[0] | depth_canny;

        cv::GaussianBlur(tt_mat, tt_mat, cv::Size(3, 3), 20.0);
//        cv::GaussianBlur(tt_mat, tt_mat, cv::Size(3, 3), 100.0);

        bool done;
        cv::Mat skel(tt_mat.size(), CV_8UC1, cv::Scalar(0));
        cv::Mat temp(tt_mat.size(), CV_8UC1);
        cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
        cv::morphologyEx(tt_mat, tt_mat, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4)));
        do
        {
            cv::morphologyEx(tt_mat, temp, cv::MORPH_OPEN, element);
            cv::bitwise_not(temp, temp);
            cv::bitwise_and(tt_mat, temp, temp);
            cv::bitwise_or(skel, temp, skel);
            cv::erode(tt_mat, tt_mat, element);

            double max;
            cv::minMaxLoc(tt_mat, 0, &max);
            done = (max == 0);
        } while (!done);


        // cv::imshow("Mix Edges", skel);

//        free(new_buffer);
//        new_buffer = (unsigned char*)malloc(frame->width*frame->height*sizeof(int))
//        cv::imshow("Edge Data", mat);
//        cv::waitKey(0);
        std::vector<std::vector<cv::Point> > t_contours;
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;

        /// Detect edges using canny
        //Canny( src_gray, canny_output, thresh, thresh*2, 3 );
        /// Find contours
        threshold(skel, skel, 20, 255, cv::THRESH_BINARY );
        cv::findContours( skel, t_contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE, cv::Point(0, 0) );
        // ImageFrame *new_frame = new ImageFrame(frame->width, frame->height, 4*sizeof(float), new_buffer);
        // getOutQueue()->push(new_frame);

        cv::Point2d point(350, 240);
        // int contour_idx = findEnclosingContour(t_contours, hierarchy, point);

        /// Draw contours
        cv::Mat drawing = cv::Mat::zeros( mat.size(), CV_8UC3 );
        cv::circle(drawing, point, 5, cv::Scalar(10, 242, 32), cv::FILLED);
        vector<int> to_delete;
        vector<int> to_add;
        for( int i = 0; i< t_contours.size(); i++ ) {
            double area = cv::contourArea(t_contours[i]);
            if(area < 500) continue;
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
        }

        /*if(contour_idx >= 0){
            cv::Scalar color = cv::Scalar( 255, 20, 20); //rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( drawing, contours, contour_idx, color, 3, 8, hierarchy, 0, cv::Point() );

            std::vector<cv::Point> ransac_results = {performRansacOnCotour(t_image, contours, contour_idx, point, hierarchy)};
            std::vector<std::vector<cv::Point>> results;
            results.push_back(ransac_results);
            cv::fillPoly(drawing, results, cv::Scalar(32, 32, 240));
        }*/

        /// Show in a window
//        cv::namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
        cv::imshow( "Contours", drawing );
        cv::waitKey(100);

        free(new_buffer);
        getInQueue()->pop();

        /// Add edge information to frame element
        frame_element->setEdgeData(depth_canny, channels[0], drawing);

        /// Push Contour data along with frame data to next stage in pipeline
        ContourAttributes* contour_data = new ContourAttributes((*frame_element), contours);
        getOutQueue()->push(contour_data);

        glDeleteTextures(1, &textID);
    }
}

int findEnclosingContour(std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Vec4i> &hierarchy, cv::Point2d &point, int idx){
    if(idx >= 0){
        int sub_contour = hierarchy[idx][2];
        do {
            auto contour = contours[sub_contour];
            if(cv::pointPolygonTest(contour, point, false) >= 0) {
                if (hierarchy[sub_contour][2] >= 0) {
                    int selected_contour = findEnclosingContour(contours, hierarchy, point, sub_contour);
                    if (selected_contour < 0) {
                        return sub_contour;
                    } else {
                        return selected_contour;
                    }
                }
                return sub_contour;
            }
            sub_contour = hierarchy[sub_contour][0];
        } while (sub_contour >= 0);
    }else{
        for (int i = 0; i < contours.size(); i++) {
            auto contour = contours[i];
            if (cv::pointPolygonTest(contour, point, false) >= 0) {
                if (hierarchy[i][2] >= 0) {
                    int selected_contour = findEnclosingContour(contours, hierarchy, point, i);
                    if(selected_contour < 0){
                        return i;
                    }else{
                        return selected_contour;
                    }
                }
                return i;
                break;
            }
        }
    }

    return -1;
}

#ifdef WITH_CUDA
std::vector<cv::Point>
performRansacOnCotour(cv::Mat &img, std::vector<std::vector<cv::Point>> &contours, int idx, cv::Point2d &point, std::vector<cv::Vec4i> &hierarchy){
    cv::Mat contour_img(cv::Size(640, 480), CV_8UC1);

    drawContours( contour_img, contours, idx, cv::Scalar(255), cv::FILLED, 8, hierarchy, 0, cv::Point() );
    double contour_area = cv::contourArea(contours[idx]);
    int expected_points = std::min((int)(contour_area * 0.4f), 150);
    int num_points = 0;
    std::vector<double3> points(expected_points);

    cv::RNG rng(3432764);

    auto rect = cv::boundingRect(contours[idx]);
    while(num_points < expected_points){
        int x = rng.uniform(rect.x, rect.x + rect.width);
        int y = rng.uniform(rect.y, rect.y + rect.height);
        if(contour_img.at<unsigned char>(y, x) >= 200) {
            points[num_points].x = x;
            points[num_points].y = y;
            points[num_points].z = img.at<float>(y, x) * 1000;
            num_points++;
        }
    }
    std::vector<double3> ransac_points =  execute_ransac(points);
    // BOOST_LOG_TRIVIAL(info) << "Printing ransac points";
    std::vector<cv::Point> ransac_results;
    for(auto p: ransac_points){
        // BOOST_LOG_TRIVIAL(info) << " " << p.x << " " << p.y << " " << p.z;
        ransac_results.push_back(cv::Point(p.x, p.y));
    }

    return ransac_results;

}
#endif