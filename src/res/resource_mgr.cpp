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
    shdr.init(vs_path, fs_path);

    return shdr;
}

string ResMgr::getResourceDir() {
    return this->res_dir_;
}
