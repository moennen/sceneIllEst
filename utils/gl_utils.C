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
         idx.emplace_back(face.mIndices[0], face.mIndices[1], face.mIndices[2] );
      }
   }
}

void gl_utils::TriMeshBuffer::reset()
{
    if (vao_id != -1)  glDeleteVertexArrays(1, &vao_id);
    vao_id = -1;

    for (auto& vboid : vbo_ids)
    {
      if (vboid != -1 ) glDeleteBuffers(1, &vboid); 
      vboid = -1;
    }

    _nvtx = 0;
    _nfaces = 0;
}

void gl_utils::TriMeshBuffer::draw()
{
   if (!_nvtx) return;
   glBindVertexArray( vao_id );
   if ( _nfaces )
      glDrawElements( GL_TRIANGLES, _nfaces * 3, GL_UNSIGNED_INT, (void*)0 );
   else
      glDrawArrays( GL_TRIANGLES, 0, _nvtx );
}

bool gl_utils::TriMeshBuffer::load(
    const size_t nvtx,
    const glm::vec3* vtx,
    const glm::vec2* uvs,
    const glm::vec3* normals,
    const size_t nfaces,
    const glm::uvec3* idx )
{
  reset();
  
  if ( nvtx == 0 )  return true;  

  if ( vtx != nullptr ) 
  {
    glGenVertexArrays( 1, &vao_id );
    glBindVertexArray( vao_id );

    glGenBuffers( 1, &vbo_ids[0] );
    glBindBuffer( GL_ARRAY_BUFFER, vbo_ids[0] );
    glBufferData( GL_ARRAY_BUFFER, nvtx * sizeof( vec3 ), value_ptr(vtx[0]),
                  GL_STATIC_DRAW );
    glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, NULL );
    glEnableVertexAttribArray( 0 );
    _nvtx = nvtx;
  }
  else return false;

  if ( normals != nullptr ) 
  {
    glGenBuffers( 1, &vbo_ids[1] );
    glBindBuffer( GL_ARRAY_BUFFER, vbo_ids[1] );
    glBufferData( GL_ARRAY_BUFFER, nvtx * sizeof( vec3 ), value_ptr(normals[0]),
                  GL_STATIC_DRAW );
    glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, 0, NULL );
    glEnableVertexAttribArray( 1 );
  }

  if ( uvs != nullptr ) 
  {
    glGenBuffers( 1, &vbo_ids[2] );
    glBindBuffer( GL_ARRAY_BUFFER, vbo_ids[2] );
    glBufferData( GL_ARRAY_BUFFER, nvtx * sizeof( vec2 ), value_ptr(uvs[0]),
                  GL_STATIC_DRAW );
    glVertexAttribPointer( 2, 2, GL_FLOAT, GL_FALSE, 0, NULL );
    glEnableVertexAttribArray( 2 );
  }

  if (nfaces > 0)
  {
    glGenBuffers(1, &vbo_ids[3]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_ids[3]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, nfaces * sizeof(uvec3), value_ptr(idx[0]), GL_STATIC_DRAW);
    _nfaces = nfaces;
  }

  return true;
}

void gl_utils::RenderProgram::reset()
{
   if ( id != -1 ) glDeleteProgram( id );
   id = -1;
}

bool gl_utils::RenderProgram::load(const char* fragment_file_path,  const char* vertex_file_path )
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

bool gl_utils::RenderProgram::activate()
{
   if ( id != -1 )
   {
      glUseProgram( id );
      return true;
   }
   return false;
}

GLint gl_utils::RenderProgram::getUniform( const char* uniformName )
{
   if ( id != -1 )
   {
      return glGetUniformLocation( id, uniformName );
   }
   return -1;
}

gl_utils::RenderTarget::RenderTarget(const glm::uvec2 isz)
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

bool gl_utils::RenderTarget::bind(const size_t natts, GLuint* atts, GLuint* depth)
{
   if ( (natts==0) || (atts==nullptr)) return false;

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