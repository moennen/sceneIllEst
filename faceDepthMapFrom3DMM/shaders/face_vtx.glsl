#version 410

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_colour;

uniform mat4 mvp;

out vec3 colour;
out vec4 position;

void main() 
{
   colour = vertex_colour;
   vec4 position = mvp * vec4(vertex_position, 1.0);
   gl_Position = position;
}