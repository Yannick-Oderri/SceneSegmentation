//
// Created by ynk on 1/17/20.
//

#include "gl_viewport.h"

GLViewport::GLViewport(AppContext& app_context):
        app_context_(&app_context),
        cam_render_texture_(-1),
        first_mouse_(true),
        last_x_(app_context_->getWindowWidth()/2.0),
        last_y_(app_context_->getWindowHeight()/2.0f),
        delta_time_(0),
        show_stereo_camera_(false),
        window_width_(app_context.getWindowWidth()),
        window_height_(app_context.getWindowHeight()),
        window_(app_context.getGLContext())
{
    initialize(app_context);
    CreateGeometries();
}


GLViewport::~GLViewport(void)
{
}


int GLViewport::initialize(AppContext& app_context) {

    glfwSetWindowUserPointer(app_context.getGLContext(), this);
    glfwSetCursorPosCallback(app_context.getGLContext(),
                             +[]( GLFWwindow* window, double xpos, double ypos )
                             {
                                 static_cast<GLViewport*>(glfwGetWindowUserPointer(window))->on_mouse(window,xpos,ypos);
                             }
    );


    glfwSetFramebufferSizeCallback(app_context.getGLContext(),
                                   +[]( GLFWwindow* window, int width, int height )
                                   {
                                       static_cast<GLViewport*>(glfwGetWindowUserPointer(window))->on_window_resize(window, width, height);
                                   }
    );

    glfwSetScrollCallback(app_context.getGLContext(),
                          +[]( GLFWwindow* window, double xoffset, double yoffset )
                          {
                              static_cast<GLViewport*>(glfwGetWindowUserPointer(window))->on_scroll(window,xoffset,yoffset);
                          }
    );

    return 0;
}

int GLViewport::CreateGeometries() {

    camera_ = std::unique_ptr<GLCamera>(new GLCamera(glm::vec3(-0.444255,0.29527,-0.218251)));

    // create opencv render rectangle
    float quad_vertices[] = {
            // positions          // colors           // texture coords
            -0.5f,  -0.5f, 0.0f,   1.0f, 1.0f, 1.0f,   1.0f, 0.0f,   // top right
            -0.5f, -1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   1.0f, 1.0f,   // bottom right
            -1.0f, -1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   0.0f, 1.0f,   // bottom left
            -1.0f,  -0.5f, 0.0f,   1.0f, 1.0f, 1.0f,   0.0f, 0.0f    // top left
    };
    unsigned int quad_indices[] = {
            0, 1, 3, // first triangle
            1, 2, 3  // second triangle
    };

    glGenVertexArrays(1, &camera_quad_vao_);
    glGenBuffers(1, &camera_quad_vbo_);
    glGenBuffers(1, &camera_quad_ebo_);
    glBindVertexArray(camera_quad_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, camera_quad_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, camera_quad_ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quad_indices), quad_indices, GL_STATIC_DRAW);
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
    glBindVertexArray( 0 );
    cam_render_shader_ = app_context_->getResMgr()->loadShader("camera_render.vs","camera_render.fs");

    // create point cloud
    glGenVertexArrays(1, &point_cloud_vao_);
    glGenBuffers(1, &point_cloud_vbo_);
    glBindVertexArray(point_cloud_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, point_cloud_vbo_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
    glBindVertexArray( 0 );
    point_cloud_shader_ = app_context_->getResMgr()->loadShader("point_cloud.vs","point_cloud.fs");
    return 0;
}

void GLViewport::ProcessInput()
{
    if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window_, true);
    if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS)
        camera_->ProcessKeyboard(Camera_Movement::FORWARD, delta_time_);
    if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS)
        camera_->ProcessKeyboard(Camera_Movement::BACKWARD, delta_time_);
    if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS)
        camera_->ProcessKeyboard(Camera_Movement::LEFT, delta_time_);
    if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS)
        camera_->ProcessKeyboard(Camera_Movement::RIGHT, delta_time_);
    if (glfwGetKey(window_, GLFW_KEY_O) == GLFW_PRESS)
        camera_->ProcessKeyboard(Camera_Movement::UP, delta_time_);
    if (glfwGetKey(window_, GLFW_KEY_P) == GLFW_PRESS)
        camera_->ProcessKeyboard(Camera_Movement::DOWN, delta_time_);
}

int GLViewport::RenderFrame(cv::Mat camera_image,cv::Mat point_cloud) {

    if (glfwGetKey(window_, GLFW_KEY_ESCAPE ) != GLFW_PRESS)
    {
        float current_frame_time = glfwGetTime();
        delta_time_ = current_frame_time - last_frame_time_;
        last_frame_time_ = current_frame_time;
        ProcessInput();
        //populate points and colors with image colors
        std::vector<cv::Vec3f> xyz_color_points;
        for (int i = 0; i < point_cloud.rows; i++)
        {
            cv::Vec3f* pLig = (cv::Vec3f*)(point_cloud.ptr(i));
            for (int j = 0; j < point_cloud.cols ; j++, pLig++)
            {
                //if (pLig[0][2] < 10000 )
                {
                    cv::Vec3f p1(pLig[0][0], -pLig[0][1], -pLig[0][2]);
                    const double max_z = 3;
                    const double min_z = 0.1;
                    //if ((fabs(p1[2]) < max_z) && (fabs(p1[2]) > min_z))
                    {
                        xyz_color_points.push_back(p1);
                        cv::Vec3b color = camera_image.at<cv::Vec3b>(i, j);
                        float col2 = ((int)camera_image.at<cv::Vec3b>(cv::Point(j, i))[0])/255.0;
                        float col1 = ((int)camera_image.at<cv::Vec3b>(cv::Point(j, i))[1])/255.0;
                        float col0 = ((int)camera_image.at<cv::Vec3b>(cv::Point(j, i))[2])/255.0;
                        xyz_color_points.push_back(cv::Vec3f(col0,col1,col2));
                    }
                }
            }
        }
        image = camera_image;
        if (!camera_image.empty()) {
            unsigned char *data = camera_image.ptr();
            int nrChannels = camera_image.channels();
            int width = camera_image.cols;
            int height = camera_image.rows;
            if (cam_render_texture_==-1)
                glGenTextures(1, &cam_render_texture_);
            glBindTexture(GL_TEXTURE_2D, cam_render_texture_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
        }

        // render scene
        glClearColor( 0.2f, 0.2f, 0.2f, 1.0f );
        glClear( GL_COLOR_BUFFER_BIT );

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        point_cloud_shader_.use();
        glBindVertexArray( point_cloud_vao_ );
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = camera_->GetViewMatrix();
        glm::mat4 projection = glm::perspective(glm::radians(camera_->getZoom()), (float)window_width_ / (float)window_height_, 0.1f, 1000.0f);
        point_cloud_shader_.setMat4f("projection", glm::value_ptr(projection));
        point_cloud_shader_.setMat4f("view", glm::value_ptr(view));
        point_cloud_shader_.setMat4f("model", glm::value_ptr(model));

        glBindBuffer(GL_ARRAY_BUFFER, point_cloud_vbo_);
        glBufferData(GL_ARRAY_BUFFER, xyz_color_points.size()*sizeof(cv::Vec3f), &xyz_color_points.front(), GL_STREAM_DRAW);
        glPointSize(3.5);
        glDrawArrays(GL_POINTS, 0, xyz_color_points.size()/2);
        glBindVertexArray( 0 );
        glDisable(GL_BLEND);
        //glEnable(GL_DEPTH_TEST);
        if (show_stereo_camera_) {
            cam_render_shader_.use();
            glActiveTexture(GL_TEXTURE0); // activate the texture unit first before binding texture
            glBindTexture(GL_TEXTURE_2D, cam_render_texture_);
            glBindVertexArray( camera_quad_vao_ );
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            glBindVertexArray( 0 );
        }
        glfwSwapBuffers( window_ );
        glfwPollEvents( );
    }
    else {
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        return -1;
    }
    return 0;
}

void GLViewport::terminate() {
    glDeleteVertexArrays( 1, &camera_quad_vao_ );
    glDeleteBuffers( 1, &camera_quad_vbo_ );
    glDeleteVertexArrays( 1, &point_cloud_vao_ );
    glDeleteBuffers( 1, &point_cloud_vbo_ );
    glfwTerminate( );
}

void GLViewport::on_window_resize(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void GLViewport::on_mouse(GLFWwindow* window, double xpos, double ypos)
{
    if (first_mouse_)
    {
        last_x_ = xpos;
        last_y_ = ypos;
        first_mouse_ = false;
    }
    float xoffset = xpos - last_x_;
    float yoffset = last_y_ - ypos;
    last_x_ = xpos;
    last_y_ = ypos;
    camera_->ProcessMouseMovement(xoffset, yoffset);
}

void GLViewport::on_scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    camera_->ProcessMouseScroll(yoffset);
}
