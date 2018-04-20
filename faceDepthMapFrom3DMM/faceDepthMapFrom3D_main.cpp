/*!
 * *****************************************************************************
 *   \file faceDepthMapFrom3D_main.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-03-19
 *   *****************************************************************************/

#include "utils/gl_utils.h"
#include "utils/gl_utils.inline.h"
#include "utils/phSpline.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

// opengl
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// opencv
#include "utils/cv_utils.h"

// face detector
#include "externals/face/faceDetector.h"

#include <memory>

#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@faceModel     |         | face detection model }"
    "{@faceKpModel   |         | face key points detection model }"
    "{@imageA        |         | image   }"
    "{@modelA        |         | model   }"
    "{@fragmentA     |         | fshader }";

static ivec2 windowSz( 1024, 768 );

void init()
{
   glClearColor( 0.0, 0.0, 0.0, 0.0 );
   glMatrixMode( GL_PROJECTION );
   glLoadIdentity();
   glOrtho( 0.0, windowSz.x, windowSz.y, 0.0, -10000.0, 10000.0 );
   glEnable( GL_BLEND );
   glEnable( GL_TEXTURE_2D );
   glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
}

void drawCircle( float x, float y, float r, int segments, const vec4& colour )
{
   glBegin( GL_TRIANGLE_FAN );
   glColor4fv( glm::value_ptr( colour * 0.5f ) );
   glVertex2f( x, y );
   glColor4fv( value_ptr( colour ) );
   for ( int n = 0; n <= segments; ++n )
   {
      float const t = 2 * M_PI * (float)n / (float)segments;
      glVertex2f( x + sin( t ) * r, y + cos( t ) * r );
   }
   glEnd();
}

void drawRect( const vec4& rect, const vec4& colour )
{
   glBegin( GL_LINE_LOOP );
   glColor4fv( glm::value_ptr( colour ) );
   glVertex2f( rect.x, rect.y );
   glVertex2f( rect.z, rect.y );
   glVertex2f( rect.z, rect.w );
   glVertex2f( rect.x, rect.w );
   glEnd();
}

void drawFaces( const vector<vec4>& faces, const vector<vector<vec2> >& facesKp )
{
   glLineWidth( 2.5 );
   for ( size_t f = 0; f < faces.size(); ++f )
   {
      const auto colour = vec4( 0.0, 1.0, 0.0, 0.85 );
      const auto& frect = faces[f];
      drawRect( frect, colour );
      const vec2 fsz( frect.z - frect.x, frect.w - frect.y );
      const float fscale = 0.015 * length( fsz );
      if ( facesKp.size() > f )
      {
         for ( const auto& kps : facesKp[f] )
         {
            drawCircle( kps.x, kps.y, fscale, 128, colour );
         }
      }
   }

   glColor4f( 1.0, 1.0, 1.0, 1.0 );
}

void drawFaceModel( const gl_utils::TriMeshBuffer& mdFace )
{
   glColor4f( 0.0, 0.0, 1.0, 1.0 );

   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   glTranslatef(0.5 * windowSz.x, 0.5 * windowSz.y,0.0);
   glScalef(2.0,2.0,0.0);
   mdFace.draw(true);
   glPopMatrix();
   glColor4f( 1.0, 1.0, 1.0, 1.0 );
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

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }
   string imgFilenameA = parser.get<string>( "@imageA" );
   string modelFilenameA = parser.get<string>( "@modelA" );

   // load and setup the input image
   cv::Mat inputA = cv::imread( imgFilenameA.c_str() );
   if ( inputA.type() != CV_32F )
   {
      inputA.convertTo( inputA, CV_32F );
      inputA *= 1.0 / 255.0;
   }
   cv::cvtColor( inputA, inputA, cv::COLOR_BGR2RGB );

   windowSz = glm::ivec2( inputA.cols, inputA.rows );

   // SDL init
   SDL_Init( SDL_INIT_EVERYTHING );
   SDL_Window* window = SDL_CreateWindow(
       "faceDepthMapFrom3DMM",
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
   gl_utils::Texture<gl_utils::RGB_32FP> texA( glm::uvec2( inputA.cols, inputA.rows ) );
   gl_utils::uploadToTexture( texA, inputA.ptr() );

   cv::Mat map = cv::Mat::zeros( inputA.rows, inputA.cols, CV_32FC3 );
   gl_utils::Texture<gl_utils::RGB_32FP> texMap( glm::uvec2( inputA.cols, inputA.rows ) );
   gl_utils::uploadToTexture( texMap, map.ptr() );

   gl_utils::Texture<gl_utils::RGB_32FP> depthTex( glm::uvec2( inputA.cols, inputA.rows ) );
   std::vector<glm::vec2> depthCtrlPtx;
   size_t selectedDepthCtrlPtx = 0;

   // Face detector
   cv::Mat img = cv::imread( imgFilenameA.c_str() );
   cv::cvtColor( img, img, cv::COLOR_BGR2RGB );
   FaceDetector faceEngine;
   faceEngine.init(
       parser.get<string>( "@faceModel" ).c_str(), parser.get<string>( "@faceKpModel" ).c_str() );
   vector<vec4> imgFaces;
   faceEngine.getFaces( img, imgFaces );
   vector<vector<vec2> > imgFacesKps( imgFaces.size() );
   if ( !imgFaces.empty() )
   {
      faceEngine.getFacesLandmarks( img, imgFaces.size(), &imgFaces[0], &imgFacesKps[0] );
   }

   // Load 3d model
   gl_utils::TriMeshBuffer mdFaceA;
   {
      vector<uvec3> vecIdx;
      vector<vec3> vecVtx;
      vector<vec2> vecUvs;
      vector<vec3> vecNormals;
      if ( !gl_utils::loadTriangleMesh(
               modelFilenameA.c_str(), vecIdx, vecVtx, vecUvs, vecNormals ) )
      {
         std::cerr << "Cannot import model : " << modelFilenameA << std::endl;
         return -1;
      }
      mdFaceA.load(
          vecVtx.size(),
          vecVtx.empty() ? nullptr : &vecVtx[0],
          vecUvs.empty() ? nullptr : &vecUvs[0],
          vecNormals.empty() ? nullptr : &vecNormals[0],
          vecIdx.size(),
          vecIdx.empty() ? nullptr : &vecIdx[0] );
      vec3 maxVtx(-1.0,-1.0,-1.0);
      vec3 minVtx(100000.0,100000.0,1000000.0);
      for (auto v : vecVtx)
      {
        maxVtx = max(maxVtx,v);
        minVtx = min(minVtx,v);
      }
      std::cout << maxVtx.x << "," << maxVtx.y << "," << maxVtx.z  << endl;
      std::cout << minVtx.x << "," << minVtx.y << "," << minVtx.z  << endl;
   }
   /*{
    vec3 vtx[4] = { {(float)0,(float)0,(float)0,}, {(float)0 + windowSz.x,(float)0,(float)0},
                    {(float)0 + (float)windowSz.x, (float)0 + (float)windowSz.y,(float)0},
                    {(float)0,(float)0 + (float)windowSz.y,(float)0}};
    uvec3 indices[2] = { {0u,1u,2u},  // first triangle (bottom left - top left - top right)
                       {0u,2u,3u} }; 

    mdFaceA.load(4, &vtx[0],nullptr,nullptr,2,&indices[0]);
   }*/



   /*string fragFilenameA = parser.get<string>( "@fragmentA" );
   gl_utils::RenderProgram renderN;
   if (!renderN.load(fragFilenameA.c_str())) 
   {
      std::cerr << "Cannot load shaders : " << fragFilenameA << std::endl;
      return -1;
   }
   if (!renderN.activate()) 
   {
      std::cerr << "Cannot use program : " << fragFilenameA << std::endl;
      return -1;
   }
   renderN.deactivate();*/
   
   bool running = true;
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
            case SDL_KEYDOWN:
               /* Check the SDLKey values and move change the coords */
               switch ( event.key.keysym.sym )
               {
                  case SDLK_0:
                     break;
                  case SDLK_2:
                     break;
               }
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
                  case SDL_BUTTON_LEFT:
                     SDL_ShowCursor( SDL_ENABLE );
                     break;
                  case SDL_BUTTON_RIGHT:
                     break;
                  default:
                     break;
               }
               break;
            case SDL_MOUSEMOTION:
            {
               break;
            }
            break;
            case SDL_MOUSEWHEEL:
               if ( event.wheel.y == 1 )  // scroll up
               {
                  // Pull up code here!
               }
               else if ( event.wheel.y == -1 )  // scroll down
               {
                  // Pull down code here!
               }
               break;
         }
      }
      glClear( GL_COLOR_BUFFER_BIT );
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();
      draw( texA.id, texA.sz.x, texA.sz.y );
      drawFaces( imgFaces, imgFacesKps );
      glPopMatrix();
      
      drawFaceModel( mdFaceA );
      
      SDL_GL_SwapWindow( window );
      if ( 1000 / 60 > ( SDL_GetTicks() - start ) )
         SDL_Delay( 1000 / 60 - ( SDL_GetTicks() - start ) );
   }
   SDL_Quit();

   return ( 0 );
}