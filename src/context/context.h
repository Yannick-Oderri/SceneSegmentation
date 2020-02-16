//
// Created by ynki9 on 12/26/19.
//

#ifndef PROJECT_EDGE_CONTEXT_H
#define PROJECT_EDGE_CONTEXT_H

#include <GL/glew.h>    // Initialize with glewInit()
#include <GLFW/glfw3.h>
#include <string>

using namespace std;
using CudaDevice = int;

class ResMgr;

class AppContext {
public:
    class Builder;
    /**
     * Application constructor
     * @param cuda_device
     * @param window_width
     * @param window_height
     * @param window
     */
    AppContext(AppContext::Builder& ctx_builder, ResMgr* const res_mgr);


    inline CudaDevice getCudaDevice(){
        return this->cuda_device_;
    }

    /// Disable Default Constructor
    AppContext() = delete;

    /**
     * Window width
     * @return
     */
    inline GLuint getWindowWidth(){
        return this->window_width_;
    }

    /**
     * Rendering window height
     * @return
     */
    inline GLuint  getWindowHeight(){
        return this->window_height_;
    }

    /**
     * GLFW Window context
     * @return
     */
    inline GLFWwindow* const getGLContext(){
        return this->window_;
    }

    /**
     * Return Resource Directory
     * @return
     */
    inline ResMgr* getResMgr(){
        return this->res_mgr_;
    }

private:
    const CudaDevice cuda_device_;
    const GLuint window_width_;
    const GLuint window_height_;
    GLFWwindow* const window_;
    ResMgr* const res_mgr_;

};


/*****************************************************/
class AppContext::Builder {
public:
    /// Default Constructor
    Builder():
    cuda_device_(-1),
    window_width_(-1),
    window_height_(-1){}

    /**
     * Builds an application context
     * @return
     */
    AppContext* const Build();

    /**
     * Check if CUDA device available
     * @return
     */
    bool isCudaInitialized(){return this->cuda_device_ != -1;}

    /**
     * Set viewport dimensions
     * @param viewport_width
     * @param viewport_height
     */
    inline void setViewPortDimensions(int viewport_width, int viewport_height){
        this->window_width_ = viewport_width;
        this->window_height_ = viewport_height;
    }

    /**
     * Set main window title
     * @param title
     */
    inline void setWindowTitle(string title){
        this->window_title_ = title;
    }

    /**
     * Set resource subdirectory
     * @param res_dir
     */
    inline void setResDir(string res_dir){
        this->res_dir_ = res_dir;
    }


public:
    friend class AppContext;
    int initializeMainWindow();
    int initializeCuda();
    int initializeResMgr();

    CudaDevice cuda_device_;
    int window_width_;
    int window_height_;
    GLFWwindow* window_;
    string window_title_;
    string res_dir_;
    ResMgr* const res_mgr_;
};


#endif //PROJECT_EDGE_CONTEXT_H
