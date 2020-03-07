//
// Created by ynki9 on 2/15/20.
//

#include "resource_mgr.h"


ResMgr::ResMgr(AppContextBuilder& builder):
res_dir_(builder.res_dir_) {}


Shader ResMgr::loadShader(string vs, string fs) {
    string shader_dir = this->res_dir_ + "/" + RES_SHADER_DIR;

    string vs_path = shader_dir + "/" + vs;
    string fs_path = shader_dir + "/" + fs;
    BOOST_LOG_TRIVIAL(info) << "Loading Vertex Shader: " << vs_path;
    BOOST_LOG_TRIVIAL(info) << "Loading Fragment Shader: " << fs_path;

    Shader shdr;
    int res = shdr.init(vs_path, fs_path);

    if(res < 0){
        BOOST_LOG_TRIVIAL(warning) << "Failed to load Shader: " << fs;
    }



    return shdr;
}

string ResMgr::getResourceDir() {
    return this->res_dir_;
}

cv::Mat ResMgr::loadColorImage(string name, int flags) {
    string image_dir = this->res_dir_ + "/" + RES_COLOR_IMG_DIR;

    string image_path = image_dir + "/" + name;
    BOOST_LOG_TRIVIAL(info) << "Loading Image: " << image_path;

    cv::Mat img = cv::imread(image_path, flags);

    if(img.empty()){
        BOOST_LOG_TRIVIAL(error) << "Image " << image_path << ": could not be loaded";
    }

    return img;
}

cv::Mat ResMgr::loadDepthImage(string name, int flags) {
    string image_dir = this->res_dir_ + "/" + RES_DEPTH_IMG_DIR;

    string image_path = image_dir + "/" + name;
    BOOST_LOG_TRIVIAL(info) << "Loading Image: " << image_path;

    cv::Mat img = cv::imread(image_path, flags);

    if(img.empty()){
        BOOST_LOG_TRIVIAL(error) << "Image " << image_path << ": could not be loaded";
    }

    return img;
}
