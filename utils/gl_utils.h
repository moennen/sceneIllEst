//*****************************************************************************/
//
// Filename gl_utils.h
//
// Copyright (c) 2017 Autodesk, Inc.
// All rights reserved.
//
// This computer source code and related instructions and comments are the
// unpublished confidential and proprietary information of Autodesk, Inc.
// and are protected under applicable copyright and trade secret law.
// They may not be disclosed to, copied or used by any third party without
// the prior written consent of Autodesk, Inc.
//*****************************************************************************/
#ifndef _UTILS_GL_UTILS_H
#define _UTILS_GL_UTILS_H

#ifdef __APPLE__
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#else
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#endif

#include <glm/glm.hpp>

#include <vector>

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
   ~Texture() { reset(); }

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
bool readbackTexture( const Texture<fmt>&, unsigned char* );

/*struct TrianglesMeshBuffer final
{
   trianglesMeshBuffer();
   ~trianglesMeshBuffer() { reset(); }

   bool load( const size_t nb,
      const size_t* idx,
      const glm::vec3* vtx, const glm::vec2* uvs, const glm::vec3* normals );

   void reset();

   GLuint id;
};*/

bool loadTriangleMesh(
    const char* filename,
    std::vector<size_t>& idx,
    std::vector<glm::vec3>& vtx,
    std::vector<glm::vec2>& uvs,
    std::vector<glm::vec3> normals );

/*
struct renderProgram final
{
   program() : id( -1 ) {}
   ~program() { reset(); }

   bool load( const char* v_shader_path, const char* f_shader_path );
   void reset();

   bool activate();
   GLint getUniform( const char* );

   GLuint id;
};
*/

/*struct RenderTarget final
{
   renderBuffer() : id( -1 ), depthId( -1 ), sz( 0, 0 ), quadIds{-1, -1} {}
   ~renderBuffer() { reset(); }

   bool create( size_t nAttachments, const glm::ivec2 sz, const bool depth = false );
   void reset();

   glm::ivec2 size() const { return sz; }

   bool bind();
   bool drawTextures(const size_t nb, const textureBuffer3x32FP*, const );
   bool read( float* buff, const size_t buffSz, size_t attachement = 0 );
   bool readDepth( float* buff, const size_t buffSz );

   GLuint id;
   GLuint depthId;
   std::vector<GLuint> attachIds;
   glm::ivec2 sz;

   GLuint quadIds[2];
};*/
}

#endif  // _UTILS_GL_UTILS_H
