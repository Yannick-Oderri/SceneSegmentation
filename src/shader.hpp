#ifndef SHADER_H
#define SHADER_H

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <glm/glm.hpp>

using namespace std;

class Shader
{
private:
    unsigned int framebufferObjects[2];   
    unsigned int framebufferTextureObjects[2];
    
public:
    /// Delete default sahder
    Shader(): ID(0){};
    Shader(string, string);
    Shader(const Shader& cpy);

    inline void operator=(Shader const * const rhs){
        this->ID = rhs->ID;
    }

    // the program ID
    unsigned int ID;
  
    // constructor reads and builds the shader
    int init(string& vertexPath, string& fragmentPath);
    // use/activate the shader
    void use();
    void enableFramebuffer(bool);
    // ------------------------------------------------------------------------
    unsigned int getFramebuferID(int index=0);

    unsigned int getFramebufferTextureID(int index=0);

    // utility uniform functions
    // ------------------------------------------------------------------------
    void setBool(const std::string &name, bool value) const;
    // ------------------------------------------------------------------------
    void setInt(const std::string &name, int value) const;
    // ------------------------------------------------------------------------
    void setFloat(const std::string &name, float value) const;
    // ------------------------------------------------------------------------
    void setVec2(const std::string &name, const glm::vec2 &value) const;
    void setVec2(const std::string &name, float x, float y) const;
    // ------------------------------------------------------------------------
    void setVec3(const std::string &name, const glm::vec3 &value) const;
    void setVec3(const std::string &name, float x, float y, float z) const;
    // ------------------------------------------------------------------------
    void setVec4(const std::string &name, const glm::vec4 &value) const;
    void setVec4(const std::string &name, float x, float y, float z, float w);
    // ------------------------------------------------------------------------
    void setMat2(const std::string &name, const glm::mat2 &mat) const;
    // ------------------------------------------------------------------------
    void setMat3(const std::string &name, const glm::mat3 &mat) const;
    // ------------------------------------------------------------------------
    void setMat4(const std::string &name, const glm::mat4 &mat) const;
    // ------------------------------------------------------------------------
    void setMat4f(const std::string &name, float* mat) const;
    // ------------------------------------------------------------------------
    void setFloatv(const string &name, float *v, int len) const;

private:
    void initializeFramebuffer(const int, int width, int height);

};
  
#endif