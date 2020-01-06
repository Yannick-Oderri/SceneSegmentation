#version 400
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoord;

// uniform highp float texelWidth; 
// uniform highp float texelHeight;

out vec2 TexCoords;
out vec4 vertexColor; // specify a color output to the fragment shader

out vec2 leftTextureCoordinate;
out vec2 rightTextureCoordinate;

out vec2 topTextureCoordinate;
out vec2 topLeftTextureCoordinate;
out vec2 topRightTextureCoordinate;

out vec2 bottomTextureCoordinate;
out vec2 bottomLeftTextureCoordinate;
out vec2 bottomRightTextureCoordinate;

void main()
{
    TexCoords = aTexCoord; 
    float texelWidth = 1.20;
    float texelHeight = 1.20;

    vec2 widthStep = vec2(texelWidth, 0.0);
    vec2 heightStep = vec2(0.0, texelHeight);
    vec2 widthHeightStep = vec2(texelWidth, texelHeight);
    vec2 widthNegativeHeightStep = vec2(texelWidth, -texelHeight);

    TexCoords = aTexCoord.xy;
    leftTextureCoordinate = aTexCoord.xy - widthStep;
    rightTextureCoordinate = aTexCoord.xy + widthStep;

    topTextureCoordinate = aTexCoord.xy - heightStep;
    topLeftTextureCoordinate = aTexCoord.xy - widthHeightStep;
    topRightTextureCoordinate = aTexCoord.xy + widthNegativeHeightStep;

    bottomTextureCoordinate = aTexCoord.xy + heightStep;
    bottomLeftTextureCoordinate = aTexCoord.xy - widthNegativeHeightStep;
    bottomRightTextureCoordinate = aTexCoord.xy + widthHeightStep;
    
    vertexColor = vec4(aColor, 1.0); // set the output variable to a dark-red color
    gl_Position = vec4(aPos, 1.0); // see how we directly give a vec3 to vec4's constructor
}