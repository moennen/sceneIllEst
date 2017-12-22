//*****************************************************************************/
//
// Filename gl_utils.C
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

#include <utils/gl_utils.h>

#include <fstream>
#include <iostream>
#include <string>

#include <glm/gtc/type_ptr.hpp>

using namespace glm;

bool gl_utils::loadTriangleMesh(
    const char* filename,
    std::vector<size_t>& idx,
    std::vector<glm::vec3>& vtx,
    std::vector<glm::vec2>& uvs,
    std::vector<glm::vec3> normals )
{
  Assimp::Importer importer;

  const aiScene* scene = importer.ReadFile(path, 0/*aiProcess_JoinIdenticalVertices | aiProcess_SortByPType*/);
  if( !scene) {
    fprintf( stderr, importer.GetErrorString());
    getchar();
    return false;
  }
  const aiMesh* mesh = scene->mMeshes[0]; // In this simple example code we always use the 1rst mesh (in OBJ files there is often only one anyway)

  // Fill vertices positions
  vertices.reserve(mesh->mNumVertices);
  for(unsigned int i=0; i<mesh->mNumVertices; i++){
    aiVector3D pos = mesh->mVertices[i];
    vertices.push_back(glm::vec3(pos.x, pos.y, pos.z));
  }

  // Fill vertices texture coordinates
  uvs.reserve(mesh->mNumVertices);
  for(unsigned int i=0; i<mesh->mNumVertices; i++){
    aiVector3D UVW = mesh->mTextureCoords[0][i]; // Assume only 1 set of UV coords; AssImp supports 8 UV sets.
    uvs.push_back(glm::vec2(UVW.x, UVW.y));
  }

  // Fill vertices normals
  normals.reserve(mesh->mNumVertices);
  for(unsigned int i=0; i<mesh->mNumVertices; i++){
    aiVector3D n = mesh->mNormals[i];
    normals.push_back(glm::vec3(n.x, n.y, n.z));
  }


  // Fill face indices
  indices.reserve(3*mesh->mNumFaces);
  for (unsigned int i=0; i<mesh->mNumFaces; i++){
    // Assume the model has only triangles.
    indices.push_back(mesh->mFaces[i].mIndices[0]);
    indices.push_back(mesh->mFaces[i].mIndices[1]);
    indices.push_back(mesh->mFaces[i].mIndices[2]);
  }
  
  // The "scene" pointer will be deleted automatically by "importer"
}

/*void gl_utils::program::reset()
{
   if ( id != -1 ) glDeleteProgram( id );
   id = -1;
}

bool gl_utils::program::load( const char* vertex_file_path, const char* fragment_file_path )
{
   // Create the shaders
   GLuint VertexShaderID = glCreateShader( GL_VERTEX_SHADER );
   GLuint FragmentShaderID = glCreateShader( GL_FRAGMENT_SHADER );

   // Read the Vertex Shader code from the file
   std::string VertexShaderCode;
   std::ifstream VertexShaderStream( vertex_file_path, std::ios::in );
   if ( VertexShaderStream.is_open() )
   {
      std::string Line = "";
      while ( getline( VertexShaderStream, Line ) ) VertexShaderCode += "\n" + Line;
      VertexShaderStream.close();
   }
   else
   {
      printf(
          "Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ "
          "!\n",
          vertex_file_path );
      getchar();
      return false;
   }

   // Read the Fragment Shader code from the file
   std::string FragmentShaderCode;
   std::ifstream FragmentShaderStream( fragment_file_path, std::ios::in );
   if ( FragmentShaderStream.is_open() )
   {
      std::string Line = "";
      while ( getline( FragmentShaderStream, Line ) ) FragmentShaderCode += "\n" + Line;
      FragmentShaderStream.close();
   }

   GLint Result = GL_FALSE;
   int InfoLogLength;

   // Compile Vertex Shader
   printf( "Compiling shader : %s\n", vertex_file_path );
   char const* VertexSourcePointer = VertexShaderCode.c_str();
   glShaderSource( VertexShaderID, 1, &VertexSourcePointer, NULL );
   glCompileShader( VertexShaderID );

   // Check Vertex Shader
   glGetShaderiv( VertexShaderID, GL_COMPILE_STATUS, &Result );
   glGetShaderiv( VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength );
   if ( InfoLogLength > 0 )
   {
      std::vector<char> VertexShaderErrorMessage( InfoLogLength + 1 );
      glGetShaderInfoLog( VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0] );
      printf( "%s\n", &VertexShaderErrorMessage[0] );
   }

   // Compile Fragment Shader
   printf( "Compiling shader : %s\n", fragment_file_path );
   char const* FragmentSourcePointer = FragmentShaderCode.c_str();
   glShaderSource( FragmentShaderID, 1, &FragmentSourcePointer, NULL );
   glCompileShader( FragmentShaderID );

   // Check Fragment Shader
   glGetShaderiv( FragmentShaderID, GL_COMPILE_STATUS, &Result );
   glGetShaderiv( FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength );
   if ( InfoLogLength > 0 )
   {
      std::vector<char> FragmentShaderErrorMessage( InfoLogLength + 1 );
      glGetShaderInfoLog( FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0] );
      printf( "%s\n", &FragmentShaderErrorMessage[0] );
   }

   // Link the program
   printf( "Linking program\n" );
   GLuint ProgramID = glCreateProgram();
   glAttachShader( ProgramID, VertexShaderID );
   glAttachShader( ProgramID, FragmentShaderID );
   glLinkProgram( ProgramID );

   // Check the program
   glGetProgramiv( ProgramID, GL_LINK_STATUS, &Result );
   glGetProgramiv( ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength );
   if ( InfoLogLength > 0 )
   {
      std::vector<char> ProgramErrorMessage( InfoLogLength + 1 );
      glGetProgramInfoLog( ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0] );
      printf( "%s\n", &ProgramErrorMessage[0] );
   }

   glDetachShader( ProgramID, VertexShaderID );
   glDetachShader( ProgramID, FragmentShaderID );

   glDeleteShader( VertexShaderID );
   glDeleteShader( FragmentShaderID );

   id = ProgramID;

   return id != -1;
}

bool gl_utils::program::activate()
{
   if ( id != -1 )
   {
      glUseProgram( id );
      return true;
   }
   return false;
}

GLint gl_utils::program::getUniform( const char* uniformName )
{
   if ( id != -1 )
   {
      return glGetUniformLocation( id, uniformName );
   }
   return -1;
}*/

/*bool gl_utils::renderBuffer::create( size_t nAttachments, const glm::ivec2 isz, const bool depth )
{
   if ( nAttachments >= GL_MAX_COLOR_ATTACHMENTS ) return false;

   sz = isz;

   // The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.
   glGenFramebuffers( 1, &id );
   glBindFramebuffer( GL_FRAMEBUFFER, id );

   // The texture we're going to render to
   attachIds.resize( nAttachments );
   for ( size_t a = 0; a < nAttachments; ++a )
   {
      GLuint renderedTexture;
      glGenTextures( 1, &renderedTexture );

      // "Bind" the newly created texture : all future texture functions will modify this texture
      glBindTexture( GL_TEXTURE_2D, renderedTexture );

      // Give an empty image to OpenGL ( the last "0" means "empty" )
      glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, sz.x, sz.y, 0, GL_RGBA, GL_FLOAT, 0 );

      attachIds[a] = renderedTexture;
   }

   // The depth buffer
   if ( depth )
   {
      glGenTextures( 1, &depthId );
      glBindTexture( GL_TEXTURE_2D, depthId );
      glTexImage2D(
          GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, sz.x, sz.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0 );
   }

   return true;
}

void gl_utils::renderBuffer::reset()
{
   if ( quadIds[0] != -1 )
   {
      glDeleteBuffers( 2, &quadIds[0] );
      quadIds[0] = -1;
   }

   if ( depthId != -1 )
   {
      glDeleteRenderbuffers( 1, &depthId );
      depthId = -1;
   }
   glDeleteTextures( attachIds.size(), &attachIds[0] );
   attachIds.clear();
   if ( id != -1 )
   {
      glDeleteFramebuffers( 1, &id );
      id = -1;
   }
   sz = glm::ivec2( 0, 0 );
}

bool gl_utils::renderBuffer::bind()
{
   if ( id == -1 ) return false;
   glBindFramebuffer( GL_DRAW_FRAMEBUFFER, id );

   GLenum DrawBuffers[attachIds.size()];
   for ( size_t a = 0; a < attachIds.size(); ++a )
   {
      GLuint renderedTexture = attachIds[a];

      glBindTexture( GL_TEXTURE_2D, renderedTexture );
      // Poor filtering
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );

      glFramebufferTexture( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + a, renderedTexture, 0 );

      DrawBuffers[a] = GL_COLOR_ATTACHMENT0 + a;
   }

   // The depth buffer
   if ( depthId != -1 )
   {
      glBindTexture( GL_TEXTURE_2D, depthId );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
      glFramebufferTexture( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthId, 0 );
   }

   glDrawBuffers( attachIds.size(), &DrawBuffers[0] );

   // Always check that our framebuffer is ok
   if ( glCheckFramebufferStatus( GL_FRAMEBUFFER ) != GL_FRAMEBUFFER_COMPLETE ) return false;

   return true;
}

bool gl_utils::renderBuffer::draw()
{
   if ( !bind() ) return false;

   glDisable( GL_BLEND );
   glDisable( GL_SCISSOR_TEST );

   glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );

   // draw rect
   static const vec3 g_quad_vertex[6] = {vec3( -1.0f, -1.0f, 0.0f ),
                                         vec3( 1.0f, -1.0f, 0.0f ),
                                         vec3( -1.0f, 1.0f, 0.0f ),
                                         vec3( -1.0f, 1.0f, 0.0f ),
                                         vec3( 1.0f, -1.0f, 0.0f ),
                                         vec3( 1.0f, 1.0f, 0.0f )};
   static const vec2 g_quad_uv[6] = {vec2( 0.0f, 1.0f ),
                                     vec2( 1.0f, 0.0f ),
                                     vec2( 0.0f, 1.0f ),
                                     vec2( 0.0f, 1.0f ),
                                     vec2( 1.0f, 0.0f ),
                                     vec2( 1.0f, 1.0f )};

   if ( quadIds[0] == -1 )
   {
      glGenBuffers( 2, &quadIds[0] );

      glBindBuffer( GL_ARRAY_BUFFER, quadIds[0] );
      glBufferData(
          GL_ARRAY_BUFFER, sizeof( g_quad_vertex ), value_ptr( g_quad_vertex[0] ), GL_STATIC_DRAW );

      glBindBuffer( GL_ARRAY_BUFFER, quadIds[1] );
      glBufferData(
          GL_ARRAY_BUFFER, sizeof( g_quad_uv ), value_ptr( g_quad_uv[0] ), GL_STATIC_DRAW );
   }

   // 1rst attribute buffer : vertices
   glEnableVertexAttribArray( 0 );
   glBindBuffer( GL_ARRAY_BUFFER, quadIds[0] );
   glVertexAttribPointer(
       0,  // attribute 0. No particular reason for 0, but must match the layout in the shader.
       3,  // size
       GL_FLOAT,  // type
       GL_FALSE,  // normalized?
       0,         // stride
       (void*)0   // array buffer offset
   );

   glEnableVertexAttribArray( 1 );
   glBindBuffer( GL_ARRAY_BUFFER, quadIds[1] );
   glVertexAttribPointer(
       1,         // attribute
       2,         // size
       GL_FLOAT,  // type
       GL_FALSE,  // normalized?
       0,         // stride
       (void*)0   // array buffer offset
   );

   // Draw the triangles !
   glDrawArrays( GL_TRIANGLES, 0, 6 );  // 2*3 indices starting at 0 -> 2 triangles

   glDisableVertexAttribArray( 0 );
   glDisableVertexAttribArray( 1 );
}

bool gl_utils::renderBuffer::read( float* buff, const size_t buffSz, size_t attachement )
{
   if ( ( id == -1 ) || ( attachement >= attachIds.size() ) || ( attachIds[attachement] == -1 ) )
      return false;

   glNamedFramebufferReadBuffer( id, GL_COLOR_ATTACHMENT0 + attachement );
   glReadnPixels(
       0, 0, sz.x, sz.y, GL_RGBA, GL_FLOAT, buffSz * sizeof( float ), (GLvoid*)buff );

   return true;
}

bool gl_utils::renderBuffer::readDepth( float* buff, const size_t buffSz )
{
   if ( ( id == -1 ) || ( depthId == -1 ) ) return false;

   glNamedFramebufferReadBuffer( id, GL_DEPTH_ATTACHMENT );
   glReadnPixels(
       0, 0, sz.x, sz.y, GL_RGBA, GL_FLOAT, buffSz * sizeof( float ), (GLvoid*)buff );

   return true;
}
*/