/*! *****************************************************************************
 *   \file gl_utils.C
 *   \author moennen
 *   \brief
 *   \date 2018-03-16
 *   *****************************************************************************/
#include <utils/gl_utils.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>

#include <fstream>
#include <iostream>
#include <string>

#include <glm/gtc/type_ptr.hpp>

using namespace glm;

namespace
{
GLuint createShader( const char* shader_file_path, const GLenum shaderType )
{
   GLuint shaderID = -1;

   if ( shader_file_path )
   {
      // Read the Vertex Shader code from the file
      std::string shaderCode;
      std::ifstream shaderStream( shader_file_path, std::ios::in );
      if ( shaderStream.is_open() )
      {
         std::string strLine = "";
         while ( getline( shaderStream, strLine ) ) shaderCode += "\n" + strLine;
         shaderStream.close();
      }
      else
      {
         std::cerr << "Cannot open shader : " << shader_file_path << std::endl;
         return shaderID;
      }

      shaderID = glCreateShader( shaderType );

      char const* sourcePtr = shaderCode.c_str();
      glShaderSource( shaderID, 1, &sourcePtr, NULL );
      glCompileShader( shaderID );

      GLint result = GL_FALSE;
      int infoLogLength;

      glGetShaderiv( shaderID, GL_COMPILE_STATUS, &result );
      glGetShaderiv( shaderID, GL_INFO_LOG_LENGTH, &infoLogLength );
      if ( infoLogLength > 0 )
      {
         std::string errorMessage( infoLogLength, '\0' );
         glGetShaderInfoLog( shaderID, infoLogLength, NULL, &errorMessage[0] );
         std::cerr << "Error compiling " << shader_file_path << " : " << errorMessage << std::endl;
         glDeleteShader( shaderID );
         return -1;
      }
   }

   return shaderID;
}
}

bool gl_utils::loadTriangleMesh(
    const char* filename,
    std::vector<glm::uvec3>& idx,
    std::vector<glm::vec3>& vtx,
    std::vector<glm::vec2>& uvs,
    std::vector<glm::vec3>& normals )
{
   Assimp::Importer importer;

   const aiScene* scene =
       importer.ReadFile( filename, 0 /*aiProcess_JoinIdenticalVertices | aiProcess_SortByPType*/ );
   if ( !scene ) return false;

   // default to the first mesh
   const aiMesh* mesh = scene->mMeshes[0];

   if ( mesh->HasPositions() )
   {
      vtx.reserve( mesh->mNumVertices );
      for ( size_t i = 0; i < mesh->mNumVertices; ++i )
      {
         const aiVector3D& pos = mesh->mVertices[i];
         vtx.emplace_back( pos.x, pos.y, pos.z );
      }
   }

   // Fill vertices texture coordinates
   if ( mesh->HasTextureCoords( 0 ) )
   {
      uvs.reserve( mesh->mNumVertices );
      for ( unsigned int i = 0; i < mesh->mNumVertices; i++ )
      {
         const aiVector3D& UVW = mesh->mTextureCoords[0][i];
         uvs.emplace_back( UVW.x, UVW.y );
      }
   }

   // Fill vertices normals
   if ( mesh->HasNormals() )
   {
      normals.reserve( mesh->mNumVertices );
      for ( unsigned int i = 0; i < mesh->mNumVertices; i++ )
      {
         const aiVector3D& n = mesh->mNormals[i];
         normals.emplace_back( n.x, n.y, n.z );
      }
   }

   // Fill face indices
   // face are assumed to be triangles
   if ( mesh->HasFaces() )
   {
      idx.reserve( mesh->mNumFaces );
      for ( unsigned int i = 0; i < mesh->mNumFaces; i++ )
      {
         const auto& face = mesh->mFaces[i];
         // Assume the model has only triangles.
         idx.emplace_back( face.mIndices[0], face.mIndices[1], face.mIndices[2] );
      }
   }

   return true;
}

void gl_utils::TriMeshBuffer::reset()
{
   if ( vao_id != -1 ) glDeleteVertexArrays( 1, &vao_id );
   vao_id = -1;

   for ( auto& vboid : vbo_ids )
   {
      if ( vboid != -1 ) glDeleteBuffers( 1, &vboid );
      vboid = -1;
   }

   _nvtx = 0;
   _nfaces = 0;
}

void gl_utils::TriMeshBuffer::draw( const bool wireframe ) const
{
   if ( !_nvtx ) return;
   glBindVertexArray( vao_id );
   if ( wireframe ) glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
   if ( _nfaces )
   {
      glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, vbo_ids[3] );
      glDrawElements( GL_TRIANGLES, _nfaces * sizeof( uvec3 ), GL_UNSIGNED_INT, (void*)0 );
      glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
   }
   else
      glDrawArrays( GL_TRIANGLES, 0, _nvtx );
   if ( wireframe ) glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
   glBindVertexArray( 0 );
}

bool gl_utils::TriMeshBuffer::load(
    const size_t nvtx,
    const glm::vec3* vtx,
    const glm::vec2* uvs,
    const glm::vec3* normals,
    const size_t nfaces,
    const glm::uvec3* idx )
{
   // reset();

   if ( nvtx == 0 ) return true;

   if ( vtx != nullptr )
   {
      glGenVertexArrays( 1, &vao_id );
      glBindVertexArray( vao_id );
      glGenBuffers( 1, &vbo_ids[0] );
      glBindBuffer( GL_ARRAY_BUFFER, vbo_ids[0] );
      glBufferData( GL_ARRAY_BUFFER, nvtx * sizeof( vec3 ), value_ptr( vtx[0] ), GL_STATIC_DRAW );
      glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, NULL );
      glEnableVertexAttribArray( 0 );
      glBindBuffer( GL_ARRAY_BUFFER, 0 );
      _nvtx = nvtx;
   }
   else
      return false;

   if ( normals != nullptr )
   {
      glGenBuffers( 1, &vbo_ids[1] );
      glBindBuffer( GL_ARRAY_BUFFER, vbo_ids[1] );
      glBufferData(
          GL_ARRAY_BUFFER, nvtx * sizeof( vec3 ), value_ptr( normals[0] ), GL_STATIC_DRAW );
      glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, 0, NULL );
      glEnableVertexAttribArray( 1 );
      glBindBuffer( GL_ARRAY_BUFFER, 0 );
   }

   if ( uvs != nullptr )
   {
      glGenBuffers( 1, &vbo_ids[2] );
      glBindBuffer( GL_ARRAY_BUFFER, vbo_ids[2] );
      glBufferData( GL_ARRAY_BUFFER, nvtx * sizeof( vec2 ), value_ptr( uvs[0] ), GL_STATIC_DRAW );
      glVertexAttribPointer( 2, 2, GL_FLOAT, GL_FALSE, 0, NULL );
      glEnableVertexAttribArray( 2 );
      glBindBuffer( GL_ARRAY_BUFFER, 0 );
   }

   if ( nfaces > 0 )
   {
      glGenBuffers( 1, &vbo_ids[3] );
      glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, vbo_ids[3] );
      glBufferData(
          GL_ELEMENT_ARRAY_BUFFER, nfaces * sizeof( uvec3 ), value_ptr( idx[0] ), GL_STATIC_DRAW );
      _nfaces = nfaces;
      glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
   }

   glBindVertexArray( 0 );

   return true;
}

void gl_utils::RenderProgram::reset()
{
   if ( _id != -1 ) glDeleteProgram( _id );
   _id = -1;
}

bool gl_utils::RenderProgram::load( const char* fragment_file_path, const char* vertex_file_path )
{
   bool success = true;

   // Create the shaders
   GLuint vtxId = createShader( vertex_file_path, GL_VERTEX_SHADER );
   GLuint fragId = createShader( fragment_file_path, GL_FRAGMENT_SHADER );

   if ( ( vtxId == -1 ) && ( fragId == -1 ) ) return false;

   // Link the program
   _id = glCreateProgram();
   if ( vtxId != -1 ) glAttachShader( _id, vtxId );
   if ( fragId != -1 ) glAttachShader( _id, fragId );
   glLinkProgram( _id );

   // Check the program
   GLint result = GL_FALSE;
   int infoLogLength;
   glGetProgramiv( _id, GL_LINK_STATUS, &result );
   glGetProgramiv( _id, GL_INFO_LOG_LENGTH, &infoLogLength );
   if ( infoLogLength > 0 )
   {
      std::string errorMessage( infoLogLength, '\0' );
      glGetProgramInfoLog( _id, infoLogLength, NULL, &errorMessage[0] );
      std::cerr << "Error linking "
                << " : " << errorMessage << std::endl;
      success = false;
   }

   if ( vtxId != -1 )
   {
      glDetachShader( _id, vtxId );
      glDeleteShader( vtxId );
   }

   if ( fragId != -1 )
   {
      glDetachShader( _id, fragId );
      glDeleteShader( fragId );
   }

   if ( !success ) reset();

   return success;
}

bool gl_utils::RenderProgram::activate()
{
   if ( _id != -1 )
   {
      glUseProgram( _id );
      return true;
   }
   return false;
}

GLint gl_utils::RenderProgram::getUniform( const char* uniformName )
{
   if ( _id != -1 )
   {
      return glGetUniformLocation( _id, uniformName );
   }
   return -1;
}

gl_utils::RenderTarget::RenderTarget( const glm::uvec2 isz )
{
   sz = isz;
   // The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.
   glGenFramebuffers( 1, &id );
}

gl_utils::RenderTarget::~RenderTarget()
{
   if ( id != -1 )
   {
      glDeleteFramebuffers( 1, &id );
      id = -1;
   }
}

bool gl_utils::RenderTarget::bind( const size_t natts, GLuint* atts, GLuint* depth )
{
   if ( ( natts == 0 ) || ( atts == nullptr ) ) return false;

   glBindFramebuffer( GL_DRAW_FRAMEBUFFER, id );

   GLenum DrawBuffers[natts];
   for ( size_t a = 0; a < natts; ++a )
   {
      GLuint renderedTexture = atts[a];

      glBindTexture( GL_TEXTURE_2D, renderedTexture );
      // Poor filtering
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );

      glFramebufferTexture( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + a, renderedTexture, 0 );

      DrawBuffers[a] = GL_COLOR_ATTACHMENT0 + a;
      glBindTexture( GL_TEXTURE_2D, 0 );
   }

   // The depth buffer
   if ( depth != nullptr )
   {
      glBindTexture( GL_TEXTURE_2D, *depth );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
      glFramebufferTexture( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, *depth, 0 );
      glBindTexture( GL_TEXTURE_2D, 0 );
   }

   glDrawBuffers( natts, &DrawBuffers[0] );

   // Always check that our framebuffer is ok
   if ( glCheckFramebufferStatus( GL_FRAMEBUFFER ) != GL_FRAMEBUFFER_COMPLETE ) return false;

   return true;
}