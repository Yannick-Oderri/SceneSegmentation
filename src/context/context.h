//
// Created by ynki9 on 12/26/19.
//

#ifndef PROJECT_EDGE_CONTEXT_H
#define PROJECT_EDGE_CONTEXT_H

#include <GLFW/glfw3.h>
#include <string>
#include <boost/log/trivial.hpp>

using namespace std;
using CudaDevice = int;

class ResMgr;
class AppContextBuilder;

class AppContext {
public:
    /**
     * Application constructor
     * @param cuda_device
     * @param window_width
     * @param window_height
     * @param window
     */
    AppContext(AppContextBuilder& ctx_builder);


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
    ResMgr* getResMgr(){
        return this->res_mgr_;
    }

private:
    const CudaDevice cuda_device_;
    const GLuint window_width_;
    const GLuint window_height_;
    GLFWwindow* const window_;
    ResMgr* const res_mgr_;
};




#endif //PROJECT_EDGE_CONTEXT_H
