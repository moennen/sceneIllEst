
uniform sampler2D rightOfTex;
uniform sampler2D leftOfTex;

uniform vec2 mtScale;

const float diffTh = 1.5;

void main()
{
   vec2 uv = gl_TexCoord[0].st;
   vec2 rMt = texture2D( rightOfTex, uv ).xy;
   vec2 lMt = -1.0*texture2D( leftOfTex, uv + rMt * mtScale ).xy;

   float diff = distance(rMt, lMt);

   gl_FragData[0] = diff > diffTh ? vec4(-1.0) : vec4(length(0.05 * ( rMt + lMt )));
   //gl_FragData[0] = vec4(0.05*vec2(diff),0.0,1.0);
}