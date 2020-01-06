#version 400

#define SIGMA 32.0
#define BSIGMA 0.40
#define MSIZE 15

out vec4 fragColor;

uniform sampler2DRect iChannel0;
uniform vec2 iResolution;
in vec4 fragCoord;

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

float normpdf(in float x, in float sigma)
{
	return 0.39894*exp(-0.5*x*x/(sigma*sigma))/sigma;
}

float normpdf3(in vec3 v, in float sigma)
{
	return 0.39894*exp(-0.5*dot(v,v)/(sigma*sigma))/sigma;
}


void main()
{
    vec2 iResolution = vec2(1, 1);
	vec3 c = texture(iChannel0, TexCoords.xy).rgb;
		
    //declare stuff
    const int kSize = (MSIZE-1)/2;
    float kernel[MSIZE];
    vec3 final_colour = vec3(0.0);
    
    //create the 1-D kernel
    float Z = 0.0;
    for (int j = 0; j <= kSize; ++j)
    {
        kernel[kSize+j] = kernel[kSize-j] = normpdf(float(j), SIGMA);
    }
    
    
    vec3 cc;
    float factor;
    float bZ = 1.0/normpdf(0.0, BSIGMA);
    //read out the texels
    for (int i=-kSize; i <= kSize; ++i)
    {
        for (int j=-kSize; j <= kSize; ++j)
        {
            cc = texture(iChannel0, (TexCoords.xy+vec2(float(i),float(j)))).rgb;
            factor = normpdf3(cc-c, BSIGMA)*bZ*kernel[kSize+j]*kernel[kSize+i];
            Z += factor;
            final_colour += factor*cc;

        }
    }
    
    
    fragColor = vec4(final_colour/Z, 1.0);
	
}