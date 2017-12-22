/*! *****************************************************************************
*   \file shLighting_main.cpp
*   \author moennen
*   \brief 
*   \date 2017-12-19
*   *****************************************************************************/

#include "utils/gl_utils.h"
#include "utils/gl_utils.inline.h"

#include "SDL/SDL.h"
#include "SDL/SDL_opengl.h"

// opengl
/*#include <GL/glew.h>
#include <glfw3.h>
GLFWwindow* window;*/
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

// opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;

const string keys =
    "{help h usage ? |         | print this message   }"
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

   // SDL init
   SDL_Init( SDL_INIT_EVERYTHING );
   SDL_Surface* screen = SDL_SetVideoMode( windowSz.x, windowSz.y, 32, SDL_SWSURFACE | SDL_OPENGL );

   // load and setup the input image
   cv::Mat inputA = cv::imread( inputFilenameA.c_str() );
   if ( inputA.type() != CV_32F )
   {
      inputA.convertTo( inputA, CV_32F );
      inputA *= 1.0 / 255.0;
   }
   cv::cvtColor( inputA, inputA, cv::COLOR_BGR2RGB );

   //
   gl_utils::Texture<gl_utils::RGB_32FP> texA( glm::uvec2( inputA.cols, inputA.rows ) );
   gl_utils::uploadToTexture(
       texA,
       inputA.ptr() );

   cv::Mat output(texA.sz.x, texA.sz.y, CV_32FC3);
   gl_utils::readbackTexture(texA, output.ptr());

   bool running = true;
   Uint32 start;
   SDL_Event event;

   init();
   while ( running )
   {
      start = SDL_GetTicks();
      draw( texA.id, texA.sz.x, texA.sz.y );
      while ( SDL_PollEvent( &event ) )
      {
         switch ( event.type )
         {
            case SDL_QUIT:
               running = false;
               break;
         }
      }
      SDL_GL_SwapBuffers();
      if ( 1000 / 60 > ( SDL_GetTicks() - start ) )
         SDL_Delay( 1000 / 60 - ( SDL_GetTicks() - start ) );
   }
   SDL_Quit();

   return ( 0 );
}
