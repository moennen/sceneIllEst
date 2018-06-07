
#define OneOverPi 0.318309886

const float vectorScale = 0.05;

uniform sampler2D tex;

vec3 hsv2rgb( in vec3 c )
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

const float mtBound = 50.0;

void main()
{
   vec2 uv = gl_TexCoord[0].st;
   vec2 motion = texture2D( tex, uv ).xy;

   float angle = atan( motion.y, motion.x ) * OneOverPi;
   float mag = vectorScale * length( motion );
   vec3 hsv = vec3( angle, 1.0, mag );
   gl_FragData[0] = vec4( hsv2rgb( hsv ), 1.0 );

   //gl_FragData[0] = vec4(abs(motion),0.0,1.0);

   //gl_FragData[0] = vec4( min(abs(motion.x), mtBound) / mtBound );   

   //gl_FragData[0] =  motion.x > 0.0 ? vec4(0.0,1.0,0.0,1.0) :  vec4(0.0,0.0,1.0,1.0);

}