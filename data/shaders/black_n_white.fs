#version 400

out vec4 FragColor;
uniform sampler2DRect dmap;
in vec2 TexCoords;

uniform mat3 vCoeffValues;


// All components are in the range [0…1], including hue.
vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

void main() {
    vec3 rgb = texture(dmap, TexCoords).rgb;
    vec3 hsv = rgb2hsv(rgb);

    float numCoeff = 9;
    hueRadius = 1 / numCoeff;
    float vHueVal[9];


    float lumCoeff = 0;

    diffVal     = min(abs(vHueVal[0] - hsv), abs(1 - hsv));
    lumCoeff    = lumCoeff + (vCoeffValues[0] * max(0, hueRadius - diffVal));
    for(int i = 0; i < 9; i++){
        int r = floor(i / 3);
        int c = mod(i, 3);
        lumCoeff = lumCoeff + (vCoeffValues[r][c] * max(0, hueRadius - abs(vHueVal[i] - hueVal)));
    }

    float t_val = mHsl[2] * (1 + lumCoeff);

  fragColor = vec4(t_val, t_val, t_val, 1.0);

}