//*****************************************************************************/
//
// Filename gl_utils.inline.h
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
#ifndef _UTILS_GL_UTILS_INLINE_H
#define _UTILS_GL_UTILS_INLINE_H

#include <utils/gl_utils.h>

#include <iostream>
#include <string>

#include <glm/gtc/type_ptr.hpp>

namespace
{
template <gl_utils::TextureFormat fmt>
inline GLenum glTextureFormat()
{
   return fmt == gl_utils::DEPTH_32FP
              ? GL_DEPTH_COMPONENT32F
              : fmt == gl_utils::MONO_32FP
                    ? GL_R32F
                    : fmt == gl_utils::RGB_32FP
                          ? GL_RGB32F
                          : fmt == gl_utils::BGR_32FP ? GL_RGB32F : GL_RGBA32F;
}

template <gl_utils::TextureFormat fmt>
inline GLenum glTextureInputFormat()
{
   return fmt == gl_utils::DEPTH_32FP
              ? GL_DEPTH_COMPONENT
              : fmt == gl_utils::MONO_32FP
                    ? GL_RED
                    : fmt == gl_utils::RGB_32FP ? GL_RGB
                                                : fmt == gl_utils::BGR_32FP ? GL_BGR : GL_RGBA;
}

template <gl_utils::TextureFormat fmt>
inline GLenum glTextureType()
{
   return GL_FLOAT;
}

void _check_gl_error( const char* file, int line )
{
   GLenum err( glGetError() );

   while ( err != GL_NO_ERROR )
   {
      std::string error;

      switch ( err )
      {
         case GL_INVALID_OPERATION:
            error = "INVALID_OPERATION";
            break;
         case GL_INVALID_ENUM:
            error = "INVALID_ENUM";
            break;
         case GL_INVALID_VALUE:
            error = "INVALID_VALUE";
            break;
         case GL_OUT_OF_MEMORY:
            error = "OUT_OF_MEMORY";
            break;
         case GL_INVALID_FRAMEBUFFER_OPERATION:
            error = "INVALID_FRAMEBUFFER_OPERATION";
            break;
      }

      std::cerr << "GL_" << error.c_str() << " - " << file << ":" << line << std::endl;
      err = glGetError();
   }
}
#define check_gl_error() _check_gl_error( __FILE__, __LINE__ )
};

template <gl_utils::TextureFormat fmt>
bool gl_utils::Texture<fmt>::create( const glm::uvec2 size )
{
   glGenTextures( 1, &id );
   sz = size;

   glBindTexture( GL_TEXTURE_2D, id );

   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );

   glTexImage2D(
       GL_TEXTURE_2D,
       0,  // MipMap
       glTextureFormat<fmt>(),
       sz.x,
       sz.y,
       0,  // Border
       glTextureInputFormat<fmt>(),
       glTextureType<fmt>(),
       nullptr );

   return id != -1;
}

template <gl_utils::TextureFormat fmt>
void gl_utils::Texture<fmt>::reset()
{
   if ( id != -1 )
   {
      glDeleteTextures( 1, &id );
      id = -1;
      sz = glm::uvec2( 0u, 0u );
   }
}

template <gl_utils::TextureFormat fmt>
bool gl_utils::uploadToTexture( Texture<fmt>& tex, const unsigned char* buff )
{
   if ( tex.id == -1 ) return false;

   glBindTexture( GL_TEXTURE_2D, tex.id );

   glTexSubImage2D(
       GL_TEXTURE_2D,
       0,  // MipMap
       0,
       0,
       tex.sz.x,
       tex.sz.y,
       glTextureInputFormat<fmt>(),
       glTextureType<fmt>(),
       reinterpret_cast<const GLvoid*>( buff ) );

   return true;
}

template <gl_utils::TextureFormat fmt>
bool gl_utils::uploadToTexture(
    Texture<fmt>& tex,
    const unsigned char* buff,
    const glm::uvec2 sz,
    const glm::uvec2 pos )
{
   if ( tex.id == -1 ) return false;

   glBindTexture( GL_TEXTURE_2D, tex.id );

   glTexSubImage2D(
       GL_TEXTURE_2D,
       0,  // MipMap
       pos.x,
       pos.y,
       sz.x,
       sz.y,
       glTextureInputFormat<fmt>(),
       glTextureType<fmt>(),
       reinterpret_cast<const GLvoid*>( buff ) );

   return true;
}

template <gl_utils::TextureFormat fmt>
bool gl_utils::readbackTexture( const Texture<fmt>& tex, unsigned char* buff )
{
   if ( tex.id == -1 ) return false;

   glBindTexture( GL_TEXTURE_2D, tex.id );

   glGetTexImage(
       GL_TEXTURE_2D,
       0,
       glTextureFormat<fmt>(),
       glTextureType<fmt>(),
       reinterpret_cast<GLvoid*>( buff ) );

   return false;
}

#endif  // _UTILS_GL_UTILS_INLINE_H