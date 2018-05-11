/*!
* *****************************************************************************
*   \file depthMapCreationHSBS_main.cpp
*   \author moennen
*   \brief
*   \date 2018-02-19
*   *****************************************************************************/

#include "utils/gl_utils.h"
#include "utils/gl_utils.inline.h"

// opencv
#include "utils/cv_utils.h"

// optical flow
#include <libopticalFlow/oclVarOpticalFlow.h>

// SDL
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

// opengl
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;


#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;

//------------------------------------------------------------------------------------------------------------
//

static ivec2 windowSz( 1024, 768 );


void init(const int w, const int h)
{
   glMatrixMode( GL_PROJECTION );
   glLoadIdentity();
   glOrtho( 0.0, w, h, 1.0, -1.0, 1.0 );
   glEnable( GL_TEXTURE_2D );
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

void drawTex( GLuint tex, size_t w, size_t h )
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
       1,
       1,
       0,
       1,
       0,
       0,
       1,
       0,
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

void splitTexHSBS(GLuint currTex, GLuint currLeftTex, GLuint currRightTex, size_t w, size_t h )
{ 
   static gl_utils::RenderProgram splitShader;
   if (splitShader._id==-1)
   {
      splitShader.load("shaders/splitHSBS.glsl");
   }
   
   splitShader.activate();
   
   glActiveTexture(GL_TEXTURE0);
   
   glBindTexture(GL_TEXTURE_2D, currTex);
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );  
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
   
   gl_utils::RenderTarget renderTarget( uvec2(w/2,h) );
   GLuint lrTex[2] = {currLeftTex, currRightTex};
   renderTarget.bind( 2, &lrTex[0] );
   
   drawTex(currTex,w/2,h);
   
   renderTarget.unbind();
   
   glBindTexture(GL_TEXTURE_2D, 0);
   
   splitShader.deactivate();
}

void ofColorTransform(GLuint inTex, GLuint outTex, size_t w, size_t h )
{ 
   static gl_utils::RenderProgram ofCtShader;
   if (ofCtShader._id==-1)
   {
      ofCtShader.load("shaders/ofColourTransform.glsl");
   }
   
   ofCtShader.activate();
   
   glActiveTexture(GL_TEXTURE0);
   
   glBindTexture(GL_TEXTURE_2D, inTex);
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );  
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
   
   gl_utils::RenderTarget renderTarget( uvec2(w,h) );
   renderTarget.bind( 1, &outTex );
   
   drawTex(inTex,w,h);
   
   renderTarget.unbind();
   
   glBindTexture(GL_TEXTURE_2D, 0);
   
   ofCtShader.deactivate();
}


void gotoFrame( VideoCapture& in, int frame ) { in.set( CV_CAP_PROP_POS_FRAMES, frame ); }

bool readFrame( VideoCapture& in, Mat& frame )
{
   in >> frame;
   if ( frame.empty() ) return false;
   if ( frame.type() != CV_32F )
   {
      frame.convertTo( frame, CV_32F );
      frame *= 1.0 / 255.0;
   }
   if ( frame.channels() < 3 )
   {
      cv::cvtColor( frame, frame, cv::COLOR_GRAY2RGBA );
   }
   else
   {
      cv::cvtColor( frame, frame, cv::COLOR_BGR2RGBA );
   }
   return true;
}

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@videoFile     |         | inpunt file          }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }

   // Open video
   const string inputName = parser.get<string>( "@videoFile" );
   VideoCapture inputVid( inputName.c_str() );
   if ( !inputVid.isOpened() )
   {
      cerr << "Cannot open file : " << inputName << endl;
      return -1;
   }
   Mat currFrame;
   if ( !readFrame( inputVid, currFrame ) )
   {
      cerr << "Video has less than 3 frames." << endl;
      return -1;
   }

   windowSz.x = currFrame.cols/2;
   windowSz.y = currFrame.rows;
   
   // SDL init
   SDL_Init( SDL_INIT_EVERYTHING );
   SDL_Window* window = SDL_CreateWindow(
       argv[0],
       SDL_WINDOWPOS_CENTERED,
       SDL_WINDOWPOS_CENTERED,
       windowSz.x,
       windowSz.y,
       SDL_WINDOW_OPENGL );
   SDL_GLContext glCtx = SDL_GL_CreateContext( window );

   const bool wait = argc > 1;

   bool running = true;
   Uint32 start;
   SDL_Event event;

   GLenum err = glewInit();
   if ( GLEW_OK != err )
   {
      fprintf( stderr, "Error: %s\n", glewGetErrorString( err ) );
      return -1;
   }

   init(windowSz.x,windowSz.y);

   bool drawRight = false;

   gl_utils::Texture<gl_utils::RGBA_32FP> currTex( glm::uvec2( currFrame.cols, currFrame.rows ) );
   gl_utils::Texture<gl_utils::RGBA_32FP> currLeftTex( glm::uvec2( currFrame.cols/2, currFrame.rows ) );
   gl_utils::Texture<gl_utils::RGBA_32FP> currRightTex( glm::uvec2( currFrame.cols/2, currFrame.rows ) );
   
   gl_utils::uploadToTexture( currTex, currFrame.ptr() ); 

   splitTexHSBS(currTex.id, currLeftTex.id, currRightTex.id, windowSz.x*2, windowSz.y );
   
   gl_utils::Texture<gl_utils::RGBA_32FP> currOfTex( glm::uvec2( currFrame.cols/2, currFrame.rows ) );
   gl_utils::Texture<gl_utils::RGBA_32FP> prevOfTex( glm::uvec2( currFrame.cols/2, currFrame.rows ) );

   gl_utils::RenderTarget renderTarget( currOfTex.sz );
   renderTarget.bind( 1, &currOfTex.id );
   glClearColor( 0.0, 0.0, 0.0, 0.0 );
   glClear( GL_COLOR_BUFFER_BIT );
   renderTarget.unbind();

   prevOfTex.swap( currOfTex );

   // Create the optical flow estimator
   OclVarOpticalFlow ofEstimator( windowSz.x, windowSz.y, false );

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
         }
      }
      if ( running )
      {
         if ( !readFrame( inputVid, currFrame ) )
         {
            gotoFrame( inputVid, 0 );
            continue;
         }
         gl_utils::uploadToTexture( currTex, currFrame.ptr() ); 
         splitTexHSBS(currTex.id, currLeftTex.id, currRightTex.id, windowSz.x*2, windowSz.y );
   
         //prevOfTex.swap( currOfTex );
         ofEstimator.compute(
             currLeftTex.id,
             currRightTex.id,
             currLeftTex.sz.x,
             currLeftTex.sz.y,
             currOfTex.id,
             -1, //prevOfTex.id,
             1.0,
             1.0 );

         //color transform the optical flow 
         ofColorTransform(currOfTex.id, prevOfTex.id, prevOfTex.sz.x, prevOfTex.sz.y );

         draw( prevOfTex.id, prevOfTex.sz.x, prevOfTex.sz.y );
         /*if (drawRight)  draw( currRightTex.id, currRightTex.sz.x, currRightTex.sz.y );
         else draw( currLeftTex.id, currLeftTex.sz.x, currLeftTex.sz.y );*/
         drawRight = !drawRight;
         //draw( currTex.id, currTex.sz.x, currTex.sz.y );

         SDL_GL_SwapWindow( window );
         if ( 1000 / 60 > ( SDL_GetTicks() - start ) )
            SDL_Delay( 1000 / 60 - ( SDL_GetTicks() - start ) );
      }
   }

   SDL_Quit();

   inputVid.release();

   return ( 0 );
}