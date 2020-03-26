#version 400

uniform float coeff_values[6];
// Input texture
uniform sampler2DRect iChannel0;

out vec4 fragColor;

in vec2 TexCoords;


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
    vec3 rgb = texture(iChannel0, TexCoords).rgb;

    // Complete black and white conversion
    vec3 hsv = rgb2hsv(rgb);

    float num_coeff = 6.0;
    float hue_radius = 1.0 / num_coeff;
    float hue_vals[6] = float[6](0.0/6.0, 1.0/6.0, 2.0/6.0, 3.0/6.0, 4.0/6.0, 5.0/6.0);
	float lum_coeff = 0.0;

    float w_lum = 0.0;

    float diff_val = min(abs(hue_vals[0] - hsv[0]), abs(1.0 - hsv[0]));
	lum_coeff = lum_coeff + (coeff_values[0] * max(0.0, hue_radius - diff_val));
    for(int i = 0; i < coeff_values.length(); i++){
        lum_coeff = lum_coeff + (coeff_values[i] * max(0.0, hue_radius - abs(hue_vals[i] - hsv[0])));
    }

    float t_val = hsv[2] * (1.0 + lum_coeff);
    fragColor = vec4(t_val, t_val, t_val, 1.0);

}