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

vec3 log2Lin(vec3 logColour)
{
  const vec3 sg = vec3(1.0/1.055);
  const vec3 sl = vec3(1.0/12.95);
  vec3 cond = vec3(greaterThan(logColour,vec3(0.04045)));

  // if c > 0.04045  
  vec3 vg = cond * pow( (logColour + vec3(0.055))*sg, vec3(2.4) );
  // else if c <= 0.04045
  vec3 vl = (vec3(1.0) - cond) * logColour * sl;

  return (vg+vl);
}

void main() 
{
   colour = log2Lin(vertex_colour * colourOffset);
   normal = normalize((mvn*vec4(vertex_normals,1.0)).xyz);
   uv = vertex_uvs;
   position = mv * vec4(vertex_position , 1.0);
   gl_Position = mvp * vec4(vertex_position, 1.0);
}