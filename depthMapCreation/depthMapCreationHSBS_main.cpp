/*!
 * *****************************************************************************
 *   \file depthMapCreationHSBS_main.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-02-19
 *   *****************************************************************************/

#include "utils/gl_utils.h"
#include "utils/gl_utils.inline.h"
#include "utils/imgFileLst.h"

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

#include <boost/filesystem.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;
using namespace boost;

//------------------------------------------------------------------------------------------------------------
//
template <class Tex>
void draw( Tex& tex, uvec2 wSz, const bool error )
{
   static const float TexCoord[] = {0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f};
   static const GLubyte indices[] = {0, 1, 2, 0, 2, 3};

   // clear the buffer according to the error state
   glClearColor( error ? 1.0 : 0.0, error ? 0.0 : 1.0, 0.0, 1.0 );
   glClear( GL_COLOR_BUFFER_BIT );

   // set the coords
   const vec2 sz = vec2( tex.sz ) * std::min(
                                        std::min( 1.0f, (float)wSz.x / tex.sz.x ),
                                        std::min( 1.0f, (float)wSz.y / tex.sz.y ) );
   const vec2 off = 0.5f * ( vec2( wSz.x, wSz.y ) - sz );

   // Bind Texture
   glBindTexture( GL_TEXTURE_2D, tex.id );

   const vec3 Vertices[] = {vec3( off.x, off.y, 0.f ),
                            vec3( off.x + sz.x, off.y, 0.f ),
                            vec3( off.x + sz.x, off.y + sz.y, 0.f ),
                            vec3( off.x, off.y + sz.y, 0.f )};

   glEnableClientState( GL_VERTEX_ARRAY );
   glVertexPointer( 3, GL_FLOAT, 0, value_ptr( Vertices[0] ) );
   glEnableClientState( GL_TEXTURE_COORD_ARRAY );
   glTexCoordPointer( 2, GL_FLOAT, 0, TexCoord );
   glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, indices );
   glDisableClientState( GL_TEXTURE_COORD_ARRAY );
   glDisableClientState( GL_VERTEX_ARRAY );

   glBindTexture( GL_TEXTURE_2D, 0 );
}

template <class Tex>
void draw( Tex& texR, Tex& texL, uvec2 wSz, const bool error )
{
   assert( texR.sz == texL.sz );

   static const float TexCoord[] = {0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f};
   static const GLubyte indices[] = {0, 1, 2, 0, 2, 3};

   // clear the buffer according to the error state
   glClearColor( error ? 1.0 : 0.0, error ? 0.0 : 1.0, 0.0, 1.0 );
   glClear( GL_COLOR_BUFFER_BIT );

   // set the coords
   vec2 sz = vec2( texR.sz.x * 2.f, texR.sz.y );
   const float sf =
       std::min( std::min( 1.0f, (float)sz.x / wSz.x ), std::min( 1.0f, (float)sz.y / wSz.y ) );
   sz *= sf;
   vec2 off = 0.5f * ( vec2( wSz.x, wSz.y ) - sz );

   // Draw right
   glBindTexture( GL_TEXTURE_2D, texR.id );

   const vec3 VerticesR[] = {vec3( off.x, off.y, 0.f ),
                             vec3( off.x + sz.x, off.y, 0.f ),
                             vec3( off.x + sz.x, off.y + sz.y, 0.f ),
                             vec3( off.x, off.y + sz.y, 0.f )};

   glEnableClientState( GL_VERTEX_ARRAY );
   glVertexPointer( 3, GL_FLOAT, 0, value_ptr( VerticesR[0] ) );
   glEnableClientState( GL_TEXTURE_COORD_ARRAY );
   glTexCoordPointer( 2, GL_FLOAT, 0, TexCoord );
   glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, indices );
   glDisableClientState( GL_TEXTURE_COORD_ARRAY );
   glDisableClientState( GL_VERTEX_ARRAY );

   glBindTexture( GL_TEXTURE_2D, 0 );

   // Draw left
   glBindTexture( GL_TEXTURE_2D, texL.id );

   off.x += texR.sz.x * sf;

   const vec3 VerticesL[] = {vec3( off.x + sz.x, off.y, 0.f ),
                             vec3( off.x + 2.0 * sz.x, off.y, 0.f ),
                             vec3( off.x + 2.0 * sz.x, off.y + sz.y, 0.f ),
                             vec3( off.x + sz.x, off.y + sz.y, 0.f )};

   glEnableClientState( GL_VERTEX_ARRAY );
   glVertexPointer( 3, GL_FLOAT, 0, value_ptr( VerticesL[0] ) );
   glEnableClientState( GL_TEXTURE_COORD_ARRAY );
   glTexCoordPointer( 2, GL_FLOAT, 0, TexCoord );
   glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, indices );
   glDisableClientState( GL_TEXTURE_COORD_ARRAY );
   glDisableClientState( GL_VERTEX_ARRAY );

   glBindTexture( GL_TEXTURE_2D, 0 );
}

void drawQuad( size_t w, size_t h )
{
   const vec3 Vertices[] = {vec3( 0.f, 0.f, 0.f ),
                            vec3( (float)w, 0.f, 0.f ),
                            vec3( (float)w, (float)h, 0.f ),
                            vec3( 0.f, (float)h, 0.f )};
   static const GLfloat TexCoord[] = {0.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f};
   static const GLubyte indices[] = {0, 1, 2, 0, 2, 3};

   glEnableClientState( GL_VERTEX_ARRAY );
   glVertexPointer( 3, GL_FLOAT, 0, value_ptr( Vertices[0] ) );

   glEnableClientState( GL_TEXTURE_COORD_ARRAY );
   glTexCoordPointer( 2, GL_FLOAT, 0, TexCoord );

   glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, indices );

   glDisableClientState( GL_TEXTURE_COORD_ARRAY );
   glDisableClientState( GL_VERTEX_ARRAY );
}

template <class Tex>
void splitTexHSBS( Tex& currTex, Tex& currLeftTex, Tex& currRightTex )
{
   static gl_utils::RenderProgram splitShader;

   if ( splitShader._id == -1 )
   {
      splitShader.load( "shaders/splitHSBS.glsl" );
   }

   splitShader.activate();

   gl_utils::RenderTarget renderTarget( currLeftTex.sz );
   GLuint lrTex[2] = {currLeftTex.id, currRightTex.id};
   renderTarget.bind( 2, &lrTex[0] );

   glActiveTexture( GL_TEXTURE0 );
   glBindTexture( GL_TEXTURE_2D, currTex.id );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );

   drawQuad( currLeftTex.sz.x, currLeftTex.sz.y );

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

   gl_utils::RenderTarget renderTarget( uvec2( w, h ) );
   renderTarget.bind( 1, &outTex );

   glActiveTexture( GL_TEXTURE0 );

   glBindTexture( GL_TEXTURE_2D, inTex );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );

   drawQuad( w, h );

   glBindTexture( GL_TEXTURE_2D, 0 );

   renderTarget.unbind();

   ofCtShader.deactivate();
}

template <class Tex>
vec4 getTextureMean( Tex& tex )
{
   tex.generateMipmap();
   glm::vec4 toMt;
   gl_utils::readbackTexture( tex, (unsigned char*)value_ptr( toMt ), 100000 );
   return toMt;
}

template <class Tex>
void copyTexture( Tex& srcTex, Tex& dstTex )
{
   gl_utils::RenderTarget renderTarget( dstTex.sz );
   renderTarget.bind( 1, &dstTex.id );

   glActiveTexture( GL_TEXTURE0 );

   glBindTexture( GL_TEXTURE_2D, srcTex.id );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );

   drawQuad( dstTex.sz.x, dstTex.sz.y );

   glBindTexture( GL_TEXTURE_2D, 0 );

   renderTarget.unbind();
}

template <class Tex>
void computeDisparity( Tex& rlOfTex, Tex& lrOfTex, Tex& dispTex )
{
   /*static*/ gl_utils::RenderProgram sh;

   static GLint scaleMtLoc = -1;
   static GLint rightOfTexLoc = -1;
   static GLint leftOfTexLoc = -1;

   if ( sh._id == -1 )
   {
      sh.load( "shaders/disp_fwdbwd.glsl" );
      scaleMtLoc = sh.getUniform( "mtScale" );
      rightOfTexLoc = sh.getUniform( "rightOfTex" );
      leftOfTexLoc = sh.getUniform( "leftOfTex" );
   }

   sh.activate();

   gl_utils::RenderTarget renderTarget( dispTex.sz );
   renderTarget.bind( 1, &dispTex.id );

   glUniform2f( scaleMtLoc, 1.0f / dispTex.sz.x, 1.0f / dispTex.sz.y );

   glActiveTexture( GL_TEXTURE0 );
   glBindTexture( GL_TEXTURE_2D, rlOfTex.id );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
   glUniform1i( rightOfTexLoc, 0 );

   glActiveTexture( GL_TEXTURE0 + 1 );
   glBindTexture( GL_TEXTURE_2D, lrOfTex.id );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
   glUniform1i( leftOfTexLoc, 1 );

   drawQuad( dispTex.sz.x, dispTex.sz.y );

   glBindTexture( GL_TEXTURE_2D, 0 );
   glActiveTexture( GL_TEXTURE0 );
   glBindTexture( GL_TEXTURE_2D, 0 );

   renderTarget.unbind();

   sh.deactivate();
}

template <class Tex>
float computeMotionDisparity(
    OclVarOpticalFlow& ofEngine,
    Tex& rightTex,
    Tex& leftTex,
    Tex& rlOfTex,
    Tex& lrOfTex,
    Tex& dispTex )
{
   // compute right -> left motion
   ofEngine.setOpt( OclVarOpticalFlow::OptsDoRightDisparity, true );
   ofEngine.setOpt( OclVarOpticalFlow::OptsDoLeftDisparity, false );

   ofEngine.compute(
       rightTex.id, leftTex.id, rightTex.sz.x, rightTex.sz.y, rlOfTex.id, -1, 1.0, 1.0 );

   // compute left -> right motion
   ofEngine.setOpt( OclVarOpticalFlow::OptsDoRightDisparity, false );
   ofEngine.setOpt( OclVarOpticalFlow::OptsDoLeftDisparity, true );

   ofEngine.compute(
       leftTex.id, rightTex.id, leftTex.sz.x, leftTex.sz.y, lrOfTex.id, -1, 1.0, 1.0 );

   // compute the disparity
   computeDisparity( rlOfTex, lrOfTex, dispTex );

   return getTextureMean( dispTex ).x;
}

template <class Tex>
void computeFrameDivergence( Tex& frameA, Tex& frameB, Tex& mtABTex, Tex& divTex )
{
   /*static*/ gl_utils::RenderProgram sh;

   static GLint invFrameSz = -1;
   static GLint frameATexLoc = -1;
   static GLint frameBTexLoc = -1;
   static GLint mtABTexLoc = -1;

   if ( sh._id == -1 )
   {
      sh.load( "shaders/frame_divergence.glsl" );
      invFrameSz = sh.getUniform( "invFrameSz" );
      frameATexLoc = sh.getUniform( "frameA" );
      frameBTexLoc = sh.getUniform( "frameB" );
      mtABTexLoc = sh.getUniform( "mtAB" );
   }

   sh.activate();

   gl_utils::RenderTarget renderTarget( divTex.sz );
   renderTarget.bind( 1, &divTex.id );

   glUniform2f( invFrameSz, 1.0f / divTex.sz.x, 1.0f / divTex.sz.y );

   glActiveTexture( GL_TEXTURE0 );
   glBindTexture( GL_TEXTURE_2D, frameA.id );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
   glUniform1i( frameATexLoc, 0 );

   glActiveTexture( GL_TEXTURE0 + 1 );
   glBindTexture( GL_TEXTURE_2D, frameB.id );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
   glUniform1i( frameBTexLoc, 1 );

   glActiveTexture( GL_TEXTURE0 + 2 );
   glBindTexture( GL_TEXTURE_2D, mtABTex.id );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
   glUniform1i( mtABTexLoc, 1 );

   drawQuad( divTex.sz.x, divTex.sz.y );

   glBindTexture( GL_TEXTURE_2D, 0 );
   glActiveTexture( GL_TEXTURE0 + 1 );
   glBindTexture( GL_TEXTURE_2D, 0 );
   glActiveTexture( GL_TEXTURE0 );
   glBindTexture( GL_TEXTURE_2D, 0 );

   renderTarget.unbind();

   sh.deactivate();
}

template <class Tex>
float computeFrameMotionDivergence(
    OclVarOpticalFlow& ofEngine,
    Tex& fromFrame,
    Tex& toFrame,
    Tex& ofTex,
    Tex& divTex )
{
   ofEngine.compute(
       fromFrame.id, toFrame.id, fromFrame.sz.x, fromFrame.sz.y, ofTex.id, -1, 1.0, 1.0 );

   computeFrameDivergence( fromFrame, toFrame, ofTex, divTex );

   return getTextureMean( divTex ).w;
}

template <class Tex>
float updateDisparity(
    Tex& currDispTex,
    Tex& prevDispTex,
    Tex& nextDispTex,
    Tex& prevMtTex,
    Tex& nextMtTex,
    Tex& dispTex )
{
   /*static*/ gl_utils::RenderProgram sh;

   static GLint invFrameSz = -1;
   static GLint currDispTexLoc = -1;
   static GLint prevDispTexLoc = -1;
   static GLint nextDispTexLoc = -1;
   static GLint prevMtTexLoc = -1;
   static GLint nextMtTexLoc = -1;

   if ( sh._id == -1 )
   {
      sh.load( "shaders/update_disparity.glsl" );
      invFrameSz = sh.getUniform( "invFrameSz" );
      currDispTexLoc = sh.getUniform( "currDisp" );
      prevDispTexLoc = sh.getUniform( "prevDisp" );
      nextDispTexLoc = sh.getUniform( "nextDisp" );
      prevMtTexLoc = sh.getUniform( "prevMt" );
      nextMtTexLoc = sh.getUniform( "nextMt" );
   }

   sh.activate();

   gl_utils::RenderTarget renderTarget( dispTex.sz );
   renderTarget.bind( 1, &dispTex.id );

   glUniform2f( invFrameSz, 1.0f / currDispTex.sz.x, 1.0f / currDispTex.sz.y );

   glActiveTexture( GL_TEXTURE0 );
   glBindTexture( GL_TEXTURE_2D, currDispTex.id );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
   glUniform1i( currDispTexLoc, 0 );

   glActiveTexture( GL_TEXTURE0 + 1 );
   glBindTexture( GL_TEXTURE_2D, prevDispTex.id );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
   glUniform1i( prevDispTexLoc, 1 );

   glActiveTexture( GL_TEXTURE0 + 2 );
   glBindTexture( GL_TEXTURE_2D, nextDispTex.id );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
   glUniform1i( nextDispTexLoc, 2 );

   glActiveTexture( GL_TEXTURE0 + 3 );
   glBindTexture( GL_TEXTURE_2D, prevMtTex.id );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
   glUniform1i( prevMtTexLoc, 3 );

   glActiveTexture( GL_TEXTURE0 + 4 );
   glBindTexture( GL_TEXTURE_2D, nextMtTex.id );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
   glUniform1i( nextMtTexLoc, 4 );

   drawQuad( dispTex.sz.x, dispTex.sz.y );

   glBindTexture( GL_TEXTURE_2D, 0 );
   glActiveTexture( GL_TEXTURE0 + 3 );
   glBindTexture( GL_TEXTURE_2D, 0 );
   glActiveTexture( GL_TEXTURE0 + 2 );
   glBindTexture( GL_TEXTURE_2D, 0 );
   glActiveTexture( GL_TEXTURE0 + 1 );
   glBindTexture( GL_TEXTURE_2D, 0 );
   glActiveTexture( GL_TEXTURE0 );
   glBindTexture( GL_TEXTURE_2D, 0 );

   renderTarget.unbind();

   sh.deactivate();

   return getTextureMean( dispTex ).w;
}

enum class Errors
{
   Success,
   BadDisparity,
   FrameDivergence,
   DisparityDivergence,
   UploadTexture
};

void processError( Errors& err, const float delay )
{
   if ( err == Errors::Success ) return;
   std::string errorMsg;
   switch ( err )
   {
      case Errors::BadDisparity:
         errorMsg = "Bad Disparity";
         break;
      case Errors::DisparityDivergence:
         errorMsg = "Disparity Divergence";
         break;
      case Errors::FrameDivergence:
         errorMsg = "Frame Divergence";
         break;
      case Errors::UploadTexture:
         errorMsg = "Upload Texture";
         break;
      default:
         errorMsg = "Error";
   }

   cerr << "Error : " << errorMsg << flush << endl;
   err = Errors::Success;
   SDL_Delay( delay );
}

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@imgFileLst      |         | images list   }"
    "{@imgRootDir    |         | images root dir   }"
    "{@imgOutDir    |         | images output dir   }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }

   // Parameters
   uvec2 windowSz( 1920, 1200 );

   const bool toLinear = false;
   const float delayOnError = 100;

   const float gammaForDisparity = 50.0;
   const float lambdaForDisparity = 0.05;

   const float gammaForMotion = 50.0;
   const float lambdaForMotion = 0.197;

   const float minDispAmount = 0.0;
   const float maxFrameDivAmount = 1.0;
   const float maxDispDivAmount = 1.0;

   const filesystem::path outRootPath( parser.get<string>( "@imgOutDir" ) );

   // Create the list of image triplets

   ImgTripletsFileLst imgLst(
       parser.get<string>( "@imgFileLst" ).c_str(), parser.get<string>( "@imgRootDir" ).c_str() );
   if ( imgLst.size() == 0 )
   {
      cerr << "Invalid dataset : " << parser.get<string>( "@imgFileLst" ) << endl;
      return -1;
   }

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

   // Controls

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

   // Init

   GLenum err = glewInit();
   if ( GLEW_OK != err )
   {
      fprintf( stderr, "Error: %s\n", glewGetErrorString( err ) );
      return -1;
   }

   glMatrixMode( GL_PROJECTION );
   glLoadIdentity();
   glOrtho( 0.0, windowSz.x, windowSz.y, 1.0, -1.0, 1.0 );
   glEnable( GL_TEXTURE_2D );

   // Create the data
   using Img = Mat;
   using Tex = gl_utils::Texture<gl_utils::RGBA_32FP>;
   enum
   {
      WkFullTex = 0,

      FirstHalfTex = 1,
      PrevTex = 1,
      NextTex = 4,
      CurrTex = 7,

      OfRLTex = 10,
      OfLRTex = 11,

      WkHalfTex = 12,

      LastHalfTex = 12,

      DispTex = 0,
      RightTex = 1,
      LeftTex = 2,

      NumTex = 13

   };
   Tex texs[NumTex];

   // Create the optical flow estimator
   OclVarOpticalFlow::params_t ofParams = OclVarOpticalFlow::getDefaultParams();
   ofParams.nonLinearIter = 9;
   ofParams.robustIter = 4;
   ofParams.solverIter = 7;

   OclVarOpticalFlow ofEstimator( windowSz.x / 2, windowSz.y, false, ofParams );

   Errors lastError = Errors::Success;

   // Loop through the data
   for ( size_t i = 0; i < imgLst.size(); ++i )
   {
      SDL_GL_SwapWindow( window );
      processError( lastError, delayOnError );

      const auto& data = imgLst[i];

      Img img[3] = {cv_utils::imread32FC4( data._pathA, toLinear, true ),
                    cv_utils::imread32FC4( data._pathC, toLinear, true ),
                    cv_utils::imread32FC4( data._pathB, toLinear, true )};

      const uvec2 imgSz( img[0].cols, img[0].rows );
      const uvec2 hImgSz( imgSz.x / 2, imgSz.y );

      ofEstimator.setImgSize( hImgSz.x, hImgSz.y );

      // -------------- create resources

      texs[WkFullTex].create( imgSz );
      for ( size_t j = FirstHalfTex; j <= LastHalfTex; ++j ) texs[j].create( hImgSz );

      // -------------- compute disparities

      ofEstimator.setLambda( lambdaForDisparity );
      ofEstimator.setGamma( gammaForDisparity );

      for ( size_t j = PrevTex; j <= CurrTex; ++j )
      {
         gl_utils::uploadToTexture( texs[WkFullTex], img[j-1].ptr() );
         splitTexHSBS( texs[WkFullTex], texs[j + RightTex], texs[j + LeftTex] );

         Img iLeft( imgSz.y, imgSz.x/2, CV_32FC4 );
         gl_utils::readbackTexture( texs[j + LeftTex], iLeft.data );
         cvtColor( iLeft, iLeft, cv::COLOR_RGBA2BGR );
         imshow("elft",iLeft);
         waitKey();

         //draw( texs[WkFullTex], windowSz, false );
         //draw( texs[j + LeftTex], windowSz, false );
         lastError = Errors::BadDisparity;
         break;

         float valid = computeMotionDisparity(
             ofEstimator,
             texs[j + RightTex],
             texs[j + LeftTex],
             texs[OfRLTex],
             texs[OfLRTex],
             texs[j + DispTex] );

         draw( texs[j + RightTex], texs[j + LeftTex], windowSz, false );
         lastError = Errors::BadDisparity;
         break;



         if ( valid < minDispAmount )
         {
            lastError = Errors::BadDisparity;
            draw( texs[WkFullTex], windowSz, true );
            break;
         }
      }
      if ( lastError != Errors::Success ) continue;

      // -------------- estimate flows

      ofEstimator.setOpt( OclVarOpticalFlow::OptsDoRightDisparity, false );
      ofEstimator.setOpt( OclVarOpticalFlow::OptsDoLeftDisparity, false );
      ofEstimator.setLambda( lambdaForMotion );
      ofEstimator.setGamma( gammaForMotion );

      for ( size_t j = PrevTex; j <= NextTex; j += 3 )
      {
         const bool bwd( j == PrevTex );
         const float div = computeFrameMotionDivergence(
             ofEstimator,
             texs[CurrTex + RightTex],
             texs[j + RightTex],
             texs[WkHalfTex],
             texs[bwd ? OfLRTex : OfRLTex] );
         if ( div > maxFrameDivAmount )
         {
            lastError = Errors::FrameDivergence;
            draw( texs[CurrTex + RightTex], texs[j + RightTex], windowSz, true );
            break;
         }
      }
      if ( lastError != Errors::Success ) continue;

      // ------------ compute filtered disparity

      {
         float valid = updateDisparity(
             texs[CurrTex + DispTex],
             texs[PrevTex + DispTex],
             texs[NextTex + DispTex],
             texs[OfRLTex],
             texs[OfLRTex],
             texs[WkHalfTex] );
         if ( valid > maxDispDivAmount )
         {
            lastError = Errors::DisparityDivergence;
            draw( texs[PrevTex + DispTex], texs[NextTex + DispTex], windowSz, true );
         }
      }
      if ( lastError != Errors::Success ) continue;

      // ------------ output
      const string outBasename = filesystem::path( data._pathB ).stem().string();
      Mat imgOut( imgSz.y, imgSz.x, CV_32FC4 );
      for ( size_t j = DispTex; j <= RightTex; ++j )
      {
         copyTexture( texs[CurrTex + j], texs[WkFullTex] );
         if ( !gl_utils::readbackTexture( texs[WkFullTex], imgOut.data ) )
         {
            lastError = Errors::UploadTexture;
            draw( texs[CurrTex + j], windowSz, true );
            break;
         }
         cvtColor( imgOut, imgOut, cv::COLOR_RGBA2BGR );
         const filesystem::path f(
             outRootPath /
             filesystem::path( outBasename + string( j == RightTex ? "_i" : "_d" ) + ".exr" ) );
         imwrite( f.string().c_str(), imgOut );
      }
      if ( lastError != Errors::Success ) continue;

      // --------------- draw

      draw( texs[CurrTex + RightTex], texs[CurrTex + DispTex], windowSz, false );
   }

   SDL_Quit();

   return ( 0 );
}
