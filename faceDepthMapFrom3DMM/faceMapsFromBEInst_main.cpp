/*!
 * *****************************************************************************
 *   \file faceDepthMapFrom3D_main.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-03-19
 *   *****************************************************************************/

#include "utils/gl_utils.h"
#include "utils/gl_utils.inline.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

// opengl
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// opencv
#include "utils/cv_utils.h"

// face detector / models
#include "externals/face/beFaceMModel.h"
#include "externals/face/beFaceModelInstance.h"

#include <memory>
#include <random>
#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;

static ivec2 windowSz( 1024, 768 );

void init( const int w, const int h )
{
   glMatrixMode( GL_PROJECTION );
   glLoadIdentity();
   glOrtho( 0.0, w, h, 1.0, -1.0, 1.0 );
   glEnable( GL_TEXTURE_2D );
}

void drawFaceModel( const gl_utils::TriMeshBuffer& mdFace, const glm::mat4& proj )
{
   static gl_utils::RenderProgram faceShader;
   static GLint uniMVP = -1;
   if ( faceShader._id == -1 )
   {
      faceShader.load( "./shaders/face_frag.glsl", "./shaders/face_vtx.glsl" );
      uniMVP = faceShader.getUniform( "mvp" );
   }

   faceShader.activate();

   mat4 mv;
   mv = translate( mv, vec3( 0.5 * windowSz.x, 0.5 * windowSz.y, 0.0 ) );
   mv = rotate( mv, (float)( M_PI / 180.0 ), vec3( 0.0, 1.0, 0.0 ) );

   const mat4 mvp = proj * mv;

   glUniformMatrix4fv( uniMVP, 1, 0, value_ptr( mvp ) );

   mdFace.draw();

   faceShader.deactivate();
}

void draw( GLuint tex, size_t w, size_t h )
{
   // Bind Texture
   glBindTexture( GL_TEXTURE_2D, tex );

   GLfloat Vertices[] = {(float)0,
                         (float)0,
                         0,
                         (float)0 + w,
                         (float)0,
                         0,
                         (float)0 + (float)w,
                         (float)0 + (float)h,
                         0,
                         (float)0,
                         (float)0 + (float)h,
                         0};
   GLfloat TexCoord[] = {
       0,
       0,
       1,
       0,
       1,
       1,
       0,
       1,
   };
   GLubyte indices[] = {0,
                        1,
                        2,  // first triangle (bottom left - top left - top right)
                        0,
                        2,
                        3};  // second triangle (bottom left - top right - bottom right)

   glEnableClientState( GL_VERTEX_ARRAY );
   glVertexPointer( 3, GL_FLOAT, 0, Vertices );

   glEnableClientState( GL_TEXTURE_COORD_ARRAY );
   glTexCoordPointer( 2, GL_FLOAT, 0, TexCoord );

   glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, indices );

   glDisableClientState( GL_TEXTURE_COORD_ARRAY );
   glDisableClientState( GL_VERTEX_ARRAY );

   glBindTexture( GL_TEXTURE_2D, 0 );
}

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@modelBin      |         | bin face model }"
    "{@image         |         | image   }"
    "{@modelInstance |         | bin face model instance }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }
   const string& BEfaceModelBin = parser.get<string>( "@modelBin" );
   const string& imgFilename = parser.get<string>( "@image" );
   const string& modelInstanceFilename = parser.get<string>( "@modelInstance" );

   // load and setup the input image
   cv::Mat faceImg = cv_utils::imread32FC3( imgFilename, true );
   cv::cvtColor( faceImg, faceImg, cv::COLOR_BGR2RGB );
   windowSz = glm::ivec2( faceImg.cols, faceImg.rows );

   // SDL init
   SDL_Init( SDL_INIT_EVERYTHING );
   SDL_Window* window = SDL_CreateWindow(
       "createFaceMapsFromBEInst",
       SDL_WINDOWPOS_CENTERED,
       SDL_WINDOWPOS_CENTERED,
       windowSz.x,
       windowSz.y,
       SDL_WINDOW_OPENGL );
   SDL_GLContext glCtx = SDL_GL_CreateContext( window );

   // GLEW init
   GLenum err = glewInit();
   if ( GLEW_OK != err )
   {
      fprintf( stderr, "Error: %s\n", glewGetErrorString( err ) );
      return -1;
   }

   //
   gl_utils::Texture<gl_utils::RGB_32FP> faceTex( glm::uvec2( faceImg.cols, faceImg.rows ) );
   gl_utils::uploadToTexture( faceTex, faceImg.ptr() );

   gl_utils::Texture<gl_utils::RGB_32FP> faceMapTex( glm::uvec2( faceImg.cols, faceImg.rows ) );
   gl_utils::RenderTarget renderTarget( faceMapTex.sz );

   // 3D Model
   BEFaceMModel beFaceMMd( BEfaceModelBin );
   if ( !beFaceMMd.initialized() )
   {
      std::cerr << "Cannot import model : " << BEfaceModelBin << std::endl;
      return -1;
   }

   // Instance
   BEFaceMModelInstance beFaceInst( modelInstanceFilename );
   if ( !beFaceInst.initialized() )
   {
      std::cerr << "Cannot import model instance : " << modelInstanceFilename << std::endl;
      return -1;
   }

   vector<vec3> vecVtx;
   vecVtx.resize( BEFaceMModel::NumVertices, vec3( 0.0f ) );
   std::vector<vec3> vtxCol( BEFaceMModel::NumVertices, vec3( 0.0f ) );
   if ( !beFaceMMd.get(
            beFaceInst.getShapeCoeffs(),
            beFaceInst.getExpCoeffs(),
            beFaceInst.getTexCoeffs(),
            glm::value_ptr( vecVtx[0] ),
            glm::value_ptr( vtxCol[0] ) ) )
   {
      std::cerr << "Cannot get model geometry" << std::endl;
      return -1;
   }

   vector<vec2> vecUvs;
   vector<vec3> vecNormals;
   gl_utils::TriMeshBuffer mdFace;
   mdFace.load(
       vecVtx.size(),
       vecVtx.empty() ? nullptr : &vecVtx[0],
       BEFaceMModel::NumFaces,
       beFaceMMd.getFaces() );
   mdFace.loadAttrib( 3, value_ptr( vtxCol[0] ) );

   bool running = true;
   enum
   {
      TexMode,
      MapMode,
      NumModes
   };
   int mode = TexMode;

   const glm::mat4 camProj =  // glm::perspectiveFov( (float)(67.7 * M_PI / 180.0),
                              // (float)windowSz.x, (float)windowSz.y, 0.01f, 10000.0f );
       glm::ortho( 0.0f, (float)windowSz.x, 0.0f, (float)windowSz.y, -1000.0f, 1000.0f );

   Uint32 start;
   SDL_Event event;

   init( windowSz.x, windowSz.y );

   while ( running )
   {
      start = SDL_GetTicks();
      while ( SDL_PollEvent( &event ) )
      {
         switch ( event.type )
         {
            case SDL_QUIT:
               running = false;
               break;
            case SDL_MOUSEBUTTONUP:
               switch ( event.button.button )
               {
                  case SDL_BUTTON_LEFT:
                     mode = ( mode + 1 ) % NumModes;
                     break;
                  default:
                     break;
               }
               break;
         }
      }

      glClear( GL_COLOR_BUFFER_BIT );
      glClear( GL_DEPTH_BUFFER_BIT );

      if ( mode == TexMode )
      {
         draw( faceTex.id, faceTex.sz.x, faceTex.sz.y );
      }
      else
      {
         // render to texture
         renderTarget.bind( 1, &faceMapTex.id );
         drawFaceModel( mdFace, camProj );
         renderTarget.unbind();

         // draw texture
         draw( faceMapTex.id, faceMapTex.sz.x, faceMapTex.sz.y );
      }

      SDL_GL_SwapWindow( window );
      if ( 1000 / 60 > ( SDL_GetTicks() - start ) )
         SDL_Delay( 1000 / 60 - ( SDL_GetTicks() - start ) );
   }
   SDL_Quit();

   return ( 0 );
}