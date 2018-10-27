/*! *****************************************************************************
 *   \file gl_utils.h
 *   \author moennen
 *   \brief
 *   \date 2018-17-03
 *   *****************************************************************************/
#ifndef _UTILS_GL_UTILS_H
#define _UTILS_GL_UTILS_H

#ifdef __APPLE__
#include <GL/glew.h>
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#else
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/glew.h>
#include <GL/gl.h>
#endif

#include <glm/glm.hpp>

#include <vector>
#include <array>

namespace gl_utils
{
enum TextureFormat
{
   MONO_32FP,
   RGB_32FP,
   BGR_32FP,
   RGBA_32FP,
   DEPTH_32FP
};

template <TextureFormat fmt>
struct Texture final
{
   Texture() : id( -1 ), sz( 0u, 0u ) {}
   Texture( const glm::uvec2 size ) : id( -1 ), sz( 0u, 0u ) { create( size ); }
   Texture( GLuint i, glm::uvec2 size ) : id( i ), sz( size ) {}
   ~Texture() { reset(); }

   void swap( Texture<fmt>& tex )
   {
      std::swap( id, tex.id );
      std::swap( sz, tex.sz );
   }

   void generateMipmap()
   {
      glBindTexture( GL_TEXTURE_2D, id );
      glGenerateMipmap( GL_TEXTURE_2D );
      glBindTexture( GL_TEXTURE_2D, 0 );
   }

   bool create( const glm::uvec2 size );
   void reset();

   GLuint id;
   glm::uvec2 sz;
};

template <TextureFormat fmt>
bool uploadToTexture( Texture<fmt>&, const unsigned char* );

template <TextureFormat fmt>
bool uploadToTexture(
    Texture<fmt>&,
    const unsigned char*,
    const glm::uvec2 sz,
    const glm::uvec2 pos = glm::uvec2( 0u, 0u ) );

template <TextureFormat fmt>
bool readbackTexture( const Texture<fmt>&, unsigned char*, const int level = 0 );

struct TriMeshBuffer final
{
   ~TriMeshBuffer() { reset(); }

   bool load( const size_t nvtx, const glm::vec3* vtx, const size_t nfaces, const glm::uvec3* idx );

   bool loadAttrib( const size_t dim, const float* data );

   void draw( const bool wireframe = false ) const;
   void reset();

   size_t _nvtx = 0;
   size_t _nfaces = 0;

   GLuint vao_id = -1;
   std::vector<GLuint> vbo_ids;
};

TriMeshBuffer TexQuad( const glm::uvec2& sz );

void computeNormals(
    const size_t ntri,
    const glm::uvec3* idx,
    const size_t nvtx,
    const glm::vec3* vtx,
    glm::vec3* normals );

bool loadTriangleMesh(
    const char* filename,
    std::vector<glm::uvec3>& idx,
    std::vector<glm::vec3>& vtx,
    std::vector<glm::vec2>& uvs,
    std::vector<glm::vec3>& normals,
    std::vector<glm::vec4>& vcolors );

bool saveTriangleMesh(
    const char* filename,
    const size_t nfaces,
    const glm::uvec3* idx,
    const size_t nvtx,
    const glm::vec3* vtx,
    const glm::vec2* uvs,
    const glm::vec3* normals,
    const glm::vec3* vcolors );

struct RenderProgram final
{
   RenderProgram() : _id( -1 ) {}
   ~RenderProgram() { reset(); }

   bool load( const char* f_shader_path, const char* v_shader_path = nullptr );
   void reset();

   bool activate();
   bool deactivate() { glUseProgram( 0 ); }
   GLint getUniform( const char* );

   GLuint _id;
};

struct RenderTarget final
{
   RenderTarget( const glm::uvec2 sz );
   ~RenderTarget();

   bool bind( size_t nbAtt, GLuint* atts, GLuint* depth = nullptr );
   void unbind()
   {
      glBindFramebuffer( GL_FRAMEBUFFER, 0 );
      glBindTexture( GL_TEXTURE_2D, 0 );
   }

   GLuint id;
   GLuint depth_id;
   glm::uvec2 sz;
};
}

#endif  // _UTILS_GL_UTILS_H
