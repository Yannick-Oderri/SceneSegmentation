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

    vec4 bottomColor = texture(dmap, bottomTextureCoordinate);
    vec4 centerColor = texture(dmap, TexCoords);
    vec4 leftColor = texture(dmap, leftTextureCoordinate);
    vec4 rightColor = texture(dmap, rightTextureCoordinate);
    vec4 topColor = texture(dmap, topTextureCoordinate);

    vec4 dzdx = (leftColor - rightColor) / 2;
    vec4 dzdy = (topColor - bottomColor) / 2;
    vec3 direction = vec3(-dzdx[0], -dzdy[0], 1.0);
    vec3 normal = normalize(direction);


    FragColor = vec4(normal, 1.0);
} 