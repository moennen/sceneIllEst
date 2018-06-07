#version 410

uniform mat4 mvp;

in vec3 colour;
in vec4 position;

layout(location = 0) out vec4 frag_colour;

void main() 
{
   float depth = (1.0 + (position.z / position.w))*0.5;
   //frag_colour = vec4(vec3(depth),1.0);
   frag_colour = vec4(colour,1.0);
}