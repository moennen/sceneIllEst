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
using namespace glm;

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
    "{@imageA        |         | image1 for compare   }";

static ivec2 windowSz( 1024, 768 );

void init()
{
   glClearColor( 0.0, 0.0, 0.0, 0.0 );
   glMatrixMode( GL_PROJECTION );
   glLoadIdentity();
   glOrtho( 0.0, windowSz.x, windowSz.y, 1.0, -1.0, 1.0 );
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
       0, 0, 1, 0, 1, 1, 0, 1,
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
}

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }
   string inputFilenameA = parser.get<string>( "@imageA" );

   // load and setup the input image
   cv::Mat inputA = cv::imread( inputFilenameA.c_str() );
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
   cv::Mat img = cv::imread( inputFilenameA.c_str() );
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
      draw( texA.id, texA.sz.x, texA.sz.y );
      drawFaces( imgFaces, imgFacesKps );
      SDL_GL_SwapWindow( window );
      if ( 1000 / 60 > ( SDL_GetTicks() - start ) )
         SDL_Delay( 1000 / 60 - ( SDL_GetTicks() - start ) );
   }
   SDL_Quit();

   return ( 0 );
}