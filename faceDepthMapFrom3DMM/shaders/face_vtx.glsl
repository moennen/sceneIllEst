#version 410

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_colour;
layout(location = 2) in vec3 vertex_normals;
layout(location = 3) in vec2 vertex_uvs;

uniform mat4 mvp;
uniform mat4 mv;
uniform mat4 mvn;

uniform vec3 colourOffset;

out vec3 colour;
out vec4 position;
out vec3 normal;
out vec2 uv;

void main() 
{
   colour = vertex_colour + colourOffset;
   normal = normalize((mvn*vec4(vertex_normals,1.0)).xyz);
   uv = vertex_uvs;
   position = mv * vec4(vertex_position , 1.0);
   gl_Position = mvp * vec4(vertex_position, 1.0);
}