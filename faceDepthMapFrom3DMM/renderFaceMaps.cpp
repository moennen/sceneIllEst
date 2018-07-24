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

#include <memory>
#include <random>
#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;

static ivec2 windowSz( 1024, 768 );

template <typename T>
void sampleRN( const size_t d, T* data )
{
   static std::mt19937 gen;
   std::normal_distribution<T> rd;

   for ( size_t i = 0; i < d; ++i ) data[i] = rd( gen );
}

void init()
{
   glClearColor( 0.0, 0.0, 0.0, 0.0 );
   glMatrixMode( GL_PROJECTION );
   glLoadIdentity();
   // glOrtho( 0.0, windowSz.x, windowSz.y, 0.0, -10000.0, 10000.0 );
   // glEnable( GL_BLEND );
   glEnable( GL_DEPTH_TEST );
   // glEnable( GL_CULL_FACE );
   glFrontFace( GL_CW );
   glEnable( GL_TEXTURE_2D );
   glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
}

void drawFaceModel( const gl_utils::TriMeshBuffer& mdFace, const glm::mat4& proj, float ts )
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
   mv = rotate( mv, (float)( ts * M_PI / 180.0 ), vec3( 0.0, 1.0, 0.0 ) );

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
    "{@faceModel     |         | face detection model }"
    "{@nRenders      |         | number of renders    }"
    "{@outDir        |         | output directories   }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }

   // Load model
   string BEfaceModelBin = parser.get<string>( "@faceModel" );

   BEFaceMModel beFaceMMd( BEfaceModelBin.c_str(), false );

   Eigen::VectorXf beFaceParams =
       Eigen::VectorXf::Zero( 2 * BEFaceMModel::NumCoeffs + BEFaceMModel::NumExpCoeffs );

   if ( !beFaceMMd.initialized() )
   {
      vector<uvec3> vecIdx;
      vector<vec3> vecVtx;
      vector<vec2> vecUvs;
      vector<vec3> vecNormals;
      vector<vec4> vecColors;
      if ( gl_utils::loadTriangleMesh( "/tmp/beFaceMd.fbx", vecIdx, vecVtx, vecUvs, vecNormals, vecColors ) &&
           ( vecUvs.size() == BEFaceMModel::NumVertices ) )
      {
         beFaceMMd = BEFaceMModel( BEfaceModelBin.c_str(), value_ptr( vecUvs[0] ) );
      }
      if ( !beFaceMMd.initialized() )
      {
         std::cerr << "Cannot import model : " << BEfaceModelBin << std::endl;
         return -1;
      }
      if ( !beFaceMMd.save( "/tmp/beFaceMd.bin" ) )
      {
         std::cerr << "Cannot save model." << std::endl;
         return -1;
      }
   }
   else
   {
      vector<vec3> vecVtx( BEFaceMModel::NumVertices, vec3( 0.0f ) );
      vector<vec3> vtxCol( BEFaceMModel::NumVertices, vec3( 0.0f ) );
      if ( !beFaceMMd.get(
               beFaceParams.data(),
               beFaceParams.data() + BEFaceMModel::NumCoeffs,
               beFaceParams.data() + BEFaceMModel::NumCoeffs + BEFaceMModel::NumExpCoeffs,
               glm::value_ptr( vecVtx[0] ),
               glm::value_ptr( vtxCol[0] ) ) )
      {
         std::cerr << "Cannot get model geom : " << BEfaceModelBin << std::endl;
         return -1;
      }
      if ( !gl_utils::saveTriangleMesh(
               "/tmp/beFaceMd.obj",
               BEFaceMModel::NumFaces,
               beFaceMMd.getFaces(),
               vecVtx.size(),
               &vecVtx[0],
               nullptr,
               nullptr,
               &vtxCol[0] ) )
      {
         std::cerr << "Cannot export model : " << BEfaceModelBin << std::endl;
         return -1;
      }
   }

   const uvec2 renderSz = glm::uvec2( 640, 480 );

   // SDL init
   SDL_Init( SDL_INIT_EVERYTHING );
   SDL_Window* window = SDL_CreateWindow(
       "renderFaceMaps",
       SDL_WINDOWPOS_CENTERED,
       SDL_WINDOWPOS_CENTERED,
       renderSz.x,
       renderSz.y,
       SDL_WINDOW_OPENGL );
   SDL_GLContext glCtx = SDL_GL_CreateContext( window );

   // GLEW init
   GLenum err = glewInit();
   if ( GLEW_OK != err )
   {
      cerr << "Error: " << glewGetErrorString( err ) << endl;
      return -1;
   }

   bool running = true;
   int paramMode = 0;
   bool pause = false;
   bool changed = true;

   const glm::mat4 camProj =  // glm::perspectiveFov( (float)(67.7 * M_PI / 180.0),
                              // (float)windowSz.x, (float)windowSz.y, 0.01f, 10000.0f );
       glm::ortho( 0.0f, (float)windowSz.x, 0.0f, (float)windowSz.y, -1000.0f, 1000.0f );

   gl_utils::TriMeshBuffer mdFaceA;    
   vector<vec3> vecVtx( BEFaceMModel::NumVertices, vec3( 0.0f ) );
   vector<vec3> vtxCol( BEFaceMModel::NumVertices, vec3( 0.0f ) );
         

   sampleRN( BEFaceMModel::NumCoeffs, beFaceParams.bottomRows( BEFaceMModel::NumCoeffs ).data() );
   sampleRN(
       BEFaceMModel::NumExpCoeffs,
       beFaceParams.block<BEFaceMModel::NumExpCoeffs, 1>( BEFaceMModel::NumCoeffs, 0 ).data() );
   sampleRN( BEFaceMModel::NumCoeffs, beFaceParams.topRows( BEFaceMModel::NumCoeffs ).data() );

   Uint32 start;
   SDL_Event event;

   init();
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
            case SDL_MOUSEBUTTONDOWN:
               switch ( event.button.button )
               {
                  case SDL_BUTTON_LEFT:
                     SDL_ShowCursor( SDL_DISABLE );
                     break;
                  case SDL_BUTTON_RIGHT:
                     break;
                  default:
                     break;
               }
               break;
            case SDL_MOUSEBUTTONUP:
               switch ( event.button.button )
               {
                  case SDL_BUTTON_RIGHT:
                     paramMode = ( paramMode + 1 ) % 3;
                  default:
                     changed = true;
               }
               break;
         }
      }

      glClear( GL_COLOR_BUFFER_BIT );
      glClear( GL_DEPTH_BUFFER_BIT );

      static float ts = 85.0;
      static float tsInc = 1.0;
      if ( abs( ts ) >= 85.0 ) tsInc *= -1.0;

      if ( changed )
      {
         changed = false;

         if ( paramMode == 0 )
         {
            sampleRN(
                BEFaceMModel::NumCoeffs, beFaceParams.topRows( BEFaceMModel::NumCoeffs ).data() );
         }
         else if ( paramMode == 1 )
         {
            sampleRN(
                BEFaceMModel::NumExpCoeffs,
                beFaceParams.block<BEFaceMModel::NumExpCoeffs, 1>( BEFaceMModel::NumCoeffs, 0 )
                    .data() );
         }
         else
         {
            sampleRN(
                BEFaceMModel::NumCoeffs,
                beFaceParams.bottomRows( BEFaceMModel::NumCoeffs ).data() );
         }

         beFaceMMd.get(
             beFaceParams.data(),
             beFaceParams.data() + BEFaceMModel::NumCoeffs,
             beFaceParams.data() + BEFaceMModel::NumCoeffs + BEFaceMModel::NumExpCoeffs,
             glm::value_ptr( vecVtx[0] ),
             glm::value_ptr( vtxCol[0] ) );

         mdFaceA.load(
             vecVtx.size(),
             vecVtx.empty() ? nullptr : &vecVtx[0],
             BEFaceMModel::NumFaces,
             beFaceMMd.getFaces() );
         mdFaceA.loadAttrib( 3, value_ptr( vtxCol[0] ) );
      }

      drawFaceModel( mdFaceA, camProj, ts );

      if ( !pause ) ts += tsInc;

      SDL_GL_SwapWindow( window );
      if ( 1000 / 60 > ( SDL_GetTicks() - start ) )
         SDL_Delay( 1000 / 60 - ( SDL_GetTicks() - start ) );
   }
   SDL_Quit();

   return ( 0 );
}