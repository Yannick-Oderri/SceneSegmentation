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
    vec4 bottomLeftColor = texture(dmap, bottomLeftTextureCoordinate);
    vec4 bottomRightColor = texture(dmap, bottomRightTextureCoordinate);
    vec4 centerColor = texture(dmap, TexCoords);
    vec4 leftColor = texture(dmap, leftTextureCoordinate);
    vec4 rightColor = texture(dmap, rightTextureCoordinate);
    vec4 topColor = texture(dmap, topTextureCoordinate);
    vec4 topRightColor = texture(dmap, topRightTextureCoordinate);
    vec4 topLeftColor = texture(dmap, topLeftTextureCoordinate);

    vec4 resultColor;
    resultColor += topLeftColor * convolutionMatrix_x[0][0] + topColor * convolutionMatrix_x[0][1] + topRightColor * convolutionMatrix_x[0][2];
    resultColor += leftColor * convolutionMatrix_x[1][0] + centerColor * convolutionMatrix_x[1][1] + rightColor * convolutionMatrix_x[1][2];
    resultColor += bottomLeftColor * convolutionMatrix_x[2][0] + bottomColor * convolutionMatrix_x[2][1] + bottomRightColor * convolutionMatrix_x[2][2];

    // float sobel_vert = resultColor.x;

    resultColor = topLeftColor * convolutionMatrix_y[0][0] + topColor * convolutionMatrix_y[0][1] + topRightColor * convolutionMatrix_y[0][2];
    resultColor += leftColor * convolutionMatrix_y[1][0] + centerColor * convolutionMatrix_y[1][1] + rightColor * convolutionMatrix_y[1][2];
    resultColor += bottomLeftColor * convolutionMatrix_y[2][0] + bottomColor * convolutionMatrix_y[2][1] + bottomRightColor * convolutionMatrix_y[2][2];
    //resultColor = topColor * convolutionMatrix_y[0][1] + bottomColor * convolutionMatrix_y[2][1];

    float sobel_hori = resultColor.x;
    // float grad = atan(sobel_hori, sobel_vert);
    // grad  = (grad - PI);
    // grad = (grad - radians(180))/(radians(186.6));// + sobel_hori * 0.1 + sobel_vert * 0.1;

    FragColor = vec4(sobel_hori, 0, 0, 1.0);
} 