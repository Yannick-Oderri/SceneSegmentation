#version 400 
#define PI 3.1415926538
out vec4 FragColor;
uniform sampler2DRect dmap;
uniform mat3 convolutionMatrix_x;
uniform mat3 convolutionMatrix_y;

in vec4 vertexColor; // the input variable from the vertex shader (same name and same type)  
in vec2 TexCoords;

in vec2 leftTextureCoordinate;
in vec2 rightTextureCoordinate;

in vec2 topTextureCoordinate;
in vec2 topLeftTextureCoordinate;
in vec2 topRightTextureCoordinate;

in vec2 bottomTextureCoordinate;
in vec2 bottomLeftTextureCoordinate;
in vec2 bottomRightTextureCoordinate;


void main()
{
    int depth_scale = 4000;
    vec4 bottomColor = texture(dmap, bottomTextureCoordinate) * depth_scale;
    vec4 centerColor = texture(dmap, TexCoords) * depth_scale;
    vec4 leftColor = texture(dmap, leftTextureCoordinate) * depth_scale;
    vec4 rightColor = texture(dmap, rightTextureCoordinate) * depth_scale;
    vec4 topColor = texture(dmap, topTextureCoordinate) * depth_scale;

    if(centerColor[0] != 0){
        vec4 dzdx = (leftColor - rightColor) / 2;
        vec4 dzdy = (topColor - bottomColor) / 2;
        vec3 direction = vec3(-dzdx[0], -dzdy[0], 1.0);
        vec3 normal = normalize(direction) * 0.5 + 0.5;
        FragColor = vec4(normal, 1.0);
    }else{
        FragColor = vec4(0, 0, 0, 1.0);
    }
} 