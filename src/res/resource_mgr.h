//
// Created by ynki9 on 2/15/20.
//

#ifndef PROJECT_EDGE_RESOURCE_MGR_H
#define PROJECT_EDGE_RESOURCE_MGR_H

#include <string>
#include "context/context_factory.h"
#include "shader.hpp"
#include <opencv2/opencv.hpp>

using namespace std;

const string RES_SHADER_DIR = "shaders";
const string RES_DEPTH_IMG_DIR = "images/depth";
const string RES_COLOR_IMG_DIR = "images/color";

/**
 * Resource Manager
 */
class ResMgr {
public:
    /**
     * Resource manager Constructor
     * @param app_ctx application context
     */
    ResMgr(AppContextBuilder& builder);

    /**
     * Load Shader Resource
     * @param vs vertex shader
     * @param fs fragment shader
     * @return Shader Object
     */
    Shader loadShader(string vs, string fs);

    /**
     * Load Image as OpenCV Mat
     * @param name File Name
     * @param flags
     * @return
     */
    virtual cv::Mat loadColorImage(string name, int flags=cv::IMREAD_COLOR);

    /**
     * Load Image as OpenCV Mat
     * @param name File Name
     * @param flags
     * @return
     */
    virtual cv::Mat loadDepthImage(string name, int flags=cv::IMREAD_UNCHANGED);

    /**
     * Return resource directory
     * @return
     */
    string getResourceDir();
protected:
    ResMgr(){};

private:
    string res_dir_;
    AppContext* app_ctx_;

};


#endif //PROJECT_EDGE_RESOURCE_MGR_H
