//
// Created by ynki9 on 2/15/20.
//

#ifndef PROJECT_EDGE_CONTEXT_FACTORY_H
#define PROJECT_EDGE_CONTEXT_FACTORY_H

#include "context/context.h"



class AppContextBuilder {
public:
    /// Default Constructor
    AppContextBuilder():
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
    ResMgr* res_mgr_;
};


#endif //PROJECT_EDGE_CONTEXT_FACTORY_H
