//
// Created by ynki9 on 2/15/20.
//

#ifndef PROJECT_EDGE_RESOURCE_MGR_H
#define PROJECT_EDGE_RESOURCE_MGR_H

#include <string>
#include "context/context_factory.h"
#include "shader.hpp"

using namespace std;

const string RES_SHADER_DIR = "shaders";
const string RES_DEPTH_IMG_DIR = "depth";

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
     * Return resource directory
     * @return
     */
    string getResourceDir();

private:
    string res_dir_;
    AppContext* app_ctx_;

};


#endif //PROJECT_EDGE_RESOURCE_MGR_H
