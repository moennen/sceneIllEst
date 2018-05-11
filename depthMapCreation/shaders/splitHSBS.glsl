

uniform sampler2D tex;

void main()
{
   vec2 uv = gl_TexCoord[0].st*vec2(0.5,1.0);
   gl_FragData[0] = texture2D(tex,vec2(uv.x, uv.y));
   gl_FragData[1] = texture2D(tex,vec2(uv.x+0.5, uv.y));
}