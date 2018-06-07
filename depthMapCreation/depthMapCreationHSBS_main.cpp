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

void init( const int w, const int h )
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
       0,
       1,
       1,
       1,
       1,
       0,
       0,
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

void splitTexHSBS( GLuint currTex, GLuint currLeftTex, GLuint currRightTex, size_t w, size_t h )
{
   static gl_utils::RenderProgram splitShader;
   if ( splitShader._id == -1 )
   {
      splitShader.load( "shaders/splitHSBS.glsl" );
   }

   splitShader.activate();

   glActiveTexture( GL_TEXTURE0 );

   glBindTexture( GL_TEXTURE_2D, currTex );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );

   gl_utils::RenderTarget renderTarget( uvec2( w / 2, h ) );
   GLuint lrTex[2] = {currLeftTex, currRightTex};
   renderTarget.bind( 2, &lrTex[0] );

   drawTex( currTex, w / 2, h );

   renderTarget.unbind();

   glBindTexture( GL_TEXTURE_2D, 0 );

   splitShader.deactivate();
}

void ofColorTransform( GLuint inTex, GLuint outTex, size_t w, size_t h )
{
   /*static*/ gl_utils::RenderProgram ofCtShader;
   if ( ofCtShader._id == -1 )
   {
      ofCtShader.load( "shaders/ofColourTransform.glsl" );
   }

   ofCtShader.activate();

   glActiveTexture( GL_TEXTURE0 );

   glBindTexture( GL_TEXTURE_2D, inTex );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );

   gl_utils::RenderTarget renderTarget( uvec2( w, h ) );
   renderTarget.bind( 1, &outTex );

   drawTex( inTex, w, h );

   renderTarget.unbind();

   glBindTexture( GL_TEXTURE_2D, 0 );

   ofCtShader.deactivate();
}

void computeDisparity( GLuint rightOfTex, GLuint leftOfTex, GLuint outTex, size_t w, size_t h )
{

   gl_utils::RenderProgram sh;
   if ( sh._id == -1 )
   {
      sh.load( "shaders/disp_fwdbwd.glsl" );
   }

   sh.activate();

   glActiveTexture( GL_TEXTURE0 );

   glBindTexture( GL_TEXTURE_2D, rightOfTex );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );

   glActiveTexture( GL_TEXTURE0 + 1 );

   glBindTexture( GL_TEXTURE_2D, leftOfTex );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );

   glActiveTexture( GL_TEXTURE0 );

   gl_utils::RenderTarget renderTarget( uvec2( w, h ) );
   renderTarget.bind( 1, &outTex );

   drawTex( rightOfTex, w, h );

   renderTarget.unbind();

   glBindTexture( GL_TEXTURE_2D, 0 );

   sh.deactivate();
}

void gotoFrame( VideoCapture& in, int frame ) { in.set( CV_CAP_PROP_POS_FRAMES, frame ); }

bool readFrame( VideoCapture& in, Mat& frame, bool toLinear = false )
{
   in >> frame;
   if ( frame.empty() ) return false;
   if ( frame.type() != CV_32F )
   {
      frame.convertTo( frame, CV_32F );
      frame *= 1.0 / 255.0;
   }
   if ( toLinear ) cv_utils::imToLinear( frame );
   cv::GaussianBlur( frame, frame, Size( 7, 7 ), 0.75 );

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

   windowSz.x = currFrame.cols;
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

   /*SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
   SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
   SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
   SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
   SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);*/

   const bool wait = argc > 1;
   enum
   {
      DisplayMode_Right,
      DisplayMode_Left,
      DisplayMode_RightLeft_Right,
      DisplayMode_RightLeft_Left,
      DisplayMode_Disparity,
      DisplayMode_Motion,
      DisplayMode_Num
   };
   int displayMode = DisplayMode_Disparity;
   bool rightDisparity = true;

   bool running = true;
   Uint32 start;
   SDL_Event event;

   GLenum err = glewInit();
   if ( GLEW_OK != err )
   {
      fprintf( stderr, "Error: %s\n", glewGetErrorString( err ) );
      return -1;
   }

   init( windowSz.x, windowSz.y );

   gl_utils::Texture<gl_utils::RGBA_32FP> currTex( glm::uvec2( currFrame.cols, currFrame.rows ) );
   gl_utils::Texture<gl_utils::RGBA_32FP> currLeftTex(
       glm::uvec2( currFrame.cols / 2, currFrame.rows ) );
   gl_utils::Texture<gl_utils::RGBA_32FP> currRightTex(
       glm::uvec2( currFrame.cols / 2, currFrame.rows ) );

   gl_utils::uploadToTexture( currTex, currFrame.ptr() );

   splitTexHSBS( currTex.id, currLeftTex.id, currRightTex.id, windowSz.x, windowSz.y );

   gl_utils::Texture<gl_utils::RGBA_32FP> rightOfTex(
       glm::uvec2( currFrame.cols / 2, currFrame.rows ) );
   gl_utils::Texture<gl_utils::RGBA_32FP> leftOfTex(
       glm::uvec2( currFrame.cols / 2, currFrame.rows ) );

   gl_utils::Texture<gl_utils::RGBA_32FP> outTex(
       glm::uvec2( currFrame.cols / 2, currFrame.rows ) );

   // Create the optical flow estimator
   OclVarOpticalFlow::params_t ofParams = OclVarOpticalFlow::getDefaultParams();
   ofParams.nonLinearIter = 9;
   ofParams.robustIter = 4;
   ofParams.solverIter = 7;
   // ofParams.gamma = 50.0;   // gradients weight
   // ofParams.lambda = 0.05;  // smoothness

   OclVarOpticalFlow ofEstimator( windowSz.x / 2, windowSz.y, false, ofParams );

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
                     rightDisparity = !rightDisparity;
                     break;
                  case SDL_BUTTON_RIGHT:
                     break;
                  default:
                     break;
               }
               break;
            case SDL_KEYDOWN:
               /* Check the SDLKey values and move change the coords */
               switch ( event.key.keysym.sym )
               {
                  case SDLK_0:
                     break;
                  case SDLK_2:
                     break;
                  case SDLK_SPACE:
                     displayMode = ( displayMode + 1 ) % DisplayMode_Num;
                     displayMode = displayMode < DisplayMode_RightLeft_Right
                                       ? DisplayMode_RightLeft_Right
                                       : displayMode;
                     break;
               }
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
         splitTexHSBS( currTex.id, currLeftTex.id, currRightTex.id, windowSz.x, windowSz.y );

         ofEstimator.setOpt( OclVarOpticalFlow::OptsDoRightDisparity, rightDisparity );
         ofEstimator.setOpt( OclVarOpticalFlow::OptsDoLeftDisparity, !rightDisparity );

         ofEstimator.compute(
             currLeftTex.id,
             currRightTex.id,
             currLeftTex.sz.x,
             currLeftTex.sz.y,
             rightOfTex.id,
             -1,
             1.0,
             1.0 );

         if ( displayMode == DisplayMode_Disparity )
         {
            ofEstimator.setOpt( OclVarOpticalFlow::OptsDoRightDisparity, !rightDisparity );
            ofEstimator.setOpt( OclVarOpticalFlow::OptsDoLeftDisparity, rightDisparity );

            ofEstimator.compute(
                currRightTex.id,
                currLeftTex.id,
                currLeftTex.sz.x,
                currLeftTex.sz.y,
                leftOfTex.id,
                -1,
                1.0,
                1.0 );
         }

         
         switch ( displayMode )
         {
            case DisplayMode_Motion:
               ofColorTransform( rightOfTex.id, outTex.id, outTex.sz.x, outTex.sz.y );
               draw( outTex.id, outTex.sz.x * 2, outTex.sz.y );
               break;
            case DisplayMode_Disparity:
            {
               computeDisparity( rightOfTex.id, leftOfTex.id, outTex.id, outTex.sz.x, outTex.sz.y );
               draw( outTex.id, outTex.sz.x * 2, outTex.sz.y );
               // save
               /*Mat savedFrame( disparityTex.sz.y, disparityTex.sz.x, CV_32FC4 );
               Mat savedDepth( currRightTex.sz.y, currRightTex.sz.x, CV_32FC4 );
               if ( gl_utils::readbackTexture( disparityTex, savedDepth.data ) &&
                    gl_utils::readbackTexture( currRightTex, savedFrame.data )  )
               {
                  cvtColor( savedFrame, savedFrame, cv::COLOR_RGBA2BGR );
                  imwrite( "/tmp/rightMapCV.exr", savedFrame );
                  cvtColor( savedDepth, savedDepth, cv::COLOR_RGBA2BGR );
                  imwrite( "/tmp/depthMapCV.exr", savedDepth );
               }*/
            }
            break;
            case DisplayMode_RightLeft_Right:
               displayMode = DisplayMode_RightLeft_Left;
            case DisplayMode_Right:
               draw( currRightTex.id, currRightTex.sz.x * 2, currRightTex.sz.y );
               break;
            case DisplayMode_RightLeft_Left:
               displayMode = DisplayMode_RightLeft_Right;
            case DisplayMode_Left:
               draw( currLeftTex.id, currLeftTex.sz.x * 2, currLeftTex.sz.y );
               break;
         }

         SDL_GL_SwapWindow( window );
         if ( 1000 / 60 > ( SDL_GetTicks() - start ) )
            SDL_Delay( 1000 / 60 - ( SDL_GetTicks() - start ) );
      }
   }

   SDL_Quit();

   inputVid.release();

   return ( 0 );
}