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

// face detector / models
#include "externals/face/faceDetector.h"
#include "externals/face/bFaceMModel.h"
#include "externals/face/MMFaceModel/face3dMLModelTracker.h"

#include <memory>
#include <random>
#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;

static ivec2 windowSz( 1024, 768 );

static const size_t mdFace2DLibLandmarks[13] = {37, 40, 43, 46, 31, 32, 36, 34, 49, 55, 52, 58, 9};

static const size_t bFace2DLibLandmarks[13] = {37, 40, 43, 46, 31, 32, 36, 34, 49, 55, 52, 58, 9};

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
   glOrtho( 0.0, windowSz.x, windowSz.y, 0.0, -10000.0, 10000.0 );
   glEnable( GL_BLEND );
   //glEnable( GL_DEPTH_TEST );
   //glEnable( GL_CULL_FACE );
   glFrontFace( GL_CW );
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

void drawFaceModel( const gl_utils::TriMeshBuffer& mdFace, float ts )
{
   glColor4f( 0.0, 0.1, 0.85, 1.0 );

   glMatrixMode( GL_MODELVIEW );
   // glEnable( GL_LIGHTING );
   // glEnable( GL_LIGHT0 );
   // glLightfv(GL_LIGHT0,GL_AMBIENT,value_ptr(vec4(10.0,10.0,10.0,1.0)));
   float ambientLight[] = {0.2f, 0.2f, 0.2f, 1.0f};
   float diffuseLight[] = {0.8f, 0.8f, 0.8, 1.0f};
   float specularLight[] = {0.5f, 0.5f, 0.5f, 1.0f};
   float position[] = {-1.5f, 1.0f, -4.0f, 1.0f};

   // Assign created components to GL_LIGHT0
   glLightfv( GL_LIGHT0, GL_AMBIENT, ambientLight );
   glLightfv( GL_LIGHT0, GL_DIFFUSE, diffuseLight );
   glLightfv( GL_LIGHT0, GL_SPECULAR, specularLight );
   glLightfv( GL_LIGHT0, GL_POSITION, position );
   glPushMatrix();
   glTranslatef( 0.5 * windowSz.x, 0.5 * windowSz.y, 0.0 );
   glScalef( 2.0, 2.0, 0.0 );
   glRotatef( 180.0, 0.0, 0.0, 1.0 );
   glRotatef( ts, 0.0, 1.0, 0.0 );
   mdFace.draw( true );
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

   glBindTexture( GL_TEXTURE_2D, 0 );
}

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@faceModel     |         | face detection model }"
    "{@faceKpModel   |         | face key points detection model }"
    "{@imageA        |         | image   }"
    "{@MMfaceModelMean |         | mean face model }"
    "{@MMfaceModelBin  |         | bin face model }"
    "{@BfaceModelBin  |         | bin face model 2 }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }
   string imgFilenameA = parser.get<string>( "@imageA" );
   string MMfaceModelMean = parser.get<string>( "@MMfaceModelMean" );
   string MMfaceModelBin = parser.get<string>( "@MMfaceModelBin" );
   string BfaceModelBin = parser.get<string>( "@BfaceModelBin" );

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
   /*cv::cvtColor( img, img, cv::COLOR_BGR2RGB );
   FaceDetector faceEngine;
   /*faceEngine.init(
       parser.get<string>( "@faceModel" ).c_str(), parser.get<string>( "@faceKpModel" ).c_str() );
   vector<vec4> imgFaces;
   faceEngine.getFaces( img, imgFaces );
   vector<vector<vec2> > imgFacesKps( imgFaces.size() );
   if ( !imgFaces.empty() )
   {
      faceEngine.getFacesLandmarks( img, imgFaces.size(), &imgFaces[0], &imgFacesKps[0] );
   }*/

   // Load 3d model
   Face3dMLModelTracker mmMdFace( MMfaceModelMean.c_str(), MMfaceModelBin.c_str() );
   Face3dMLModelTracker::TFaceParams mmFaceParams;
   if ( !mmMdFace.initialized() )
   {
      std::cerr << "Cannot import model : " << MMfaceModelMean << " / " << MMfaceModelBin
                << std::endl;
      return -1;
   }

   BFaceMModel bFaceMMd( BfaceModelBin.c_str() );
   Eigen::VectorXf bFaceParams =
       Eigen::VectorXf::Zero( BFaceMModel::NumCoeffs + BFaceMModel::NumCoeffs );
   if ( !bFaceMMd.initialized() )
   {
      std::cerr << "Cannot import model : " << BfaceModelBin << std::endl;
      return -1;
   }
   gl_utils::TriMeshBuffer mdFaceA;
   {
      /*vector<uvec3> vecIdx;
      vector<vec3> vecVtx;
      vector<vec2> vecUvs;
      vector<vec3> vecNormals;
      vector<vec4> vecColors;
      if ( !gl_utils::loadTriangleMesh(
               faceModelMean.c_str(), vecIdx, vecVtx, vecUvs, vecNormals, vecColors ) )
      {
         std::cerr << "Cannot import model : " << faceModelMean << std::endl;
         return -1;
      }
      if ( !gl_utils::saveTriangleMesh(
               faceModelBin.c_str(), vecIdx, vecVtx, vecUvs, vecNormals, vecColors ) )
      {
         std::cerr << "Cannot exportt model : " << faceModelBin << std::endl;
         return -1;
      }

      mdFaceA.load(
          vecVtx.size(),
          vecVtx.empty() ? nullptr : &vecVtx[0],
          vecUvs.empty() ? nullptr : &vecUvs[0],
          vecNormals.empty() ? nullptr : &vecNormals[0],
          vecIdx.size(),
          vecIdx.empty() ? nullptr : &vecIdx[0] );
      */
      vector<vec3> vecVtx;
      if ( !mmMdFace.getMeanNeutralParams( mmFaceParams ) ||
           !mmMdFace.getVerticesFromParams( mmFaceParams, vecVtx ) )
      {
         std::cerr << "Cannot get model geom : " << MMfaceModelMean << " / " << MMfaceModelBin
                   << std::endl;
         return -1;
      }
      const std::vector<glm::uvec3>& vecIdx = mmMdFace.getFaces();
      mdFaceA.load(
          vecVtx.size(),
          vecVtx.empty() ? nullptr : &vecVtx[0],
          vecIdx.size(),
          vecIdx.empty() ? nullptr : &vecIdx[0] );

      vecVtx.resize( BFaceMModel::NumVertices, vec3( 0.0f ) );
      std::vector<vec3> vtxCol( BFaceMModel::NumVertices, vec3( 0.0f ) );
      if ( !bFaceMMd.get(
               bFaceParams.data(),
               bFaceParams.data() + BFaceMModel::NumCoeffs,
               glm::value_ptr( vecVtx[0] ),
               glm::value_ptr( vtxCol[0] ) ) )
      {
         std::cerr << "Cannot get model geom : " << BfaceModelBin << std::endl;
         return -1;
      }
      vector<vec2> vecUvs;
      vector<vec3> vecNormals;
      if ( !gl_utils::saveTriangleMesh(
               "/tmp/baselMean.obj",
               BFaceMModel::NumFaces,
               bFaceMMd.getFaces(),
               vecVtx.size(),
               &vecVtx[0],
               nullptr,
               nullptr,
               &vtxCol[0] ) )
      {
         std::cerr << "Cannot exportt model : " << BfaceModelBin << std::endl;
         return -1;
      }
      mdFaceA.load(
          vecVtx.size(),
          vecVtx.empty() ? nullptr : &vecVtx[0],
          BFaceMModel::NumFaces,
          bFaceMMd.getFaces() );
      mdFaceA.loadAttrib( 3, value_ptr( vtxCol[0] ) );
      vec3 maxVtx( -1.0, -1.0, -1.0 );
      vec3 minVtx( 100000.0, 100000.0, 1000000.0 );
      for ( auto v : vecVtx )
      {
         maxVtx = max( maxVtx, v );
         minVtx = min( minVtx, v );
      }
      std::cout << maxVtx.x << "," << maxVtx.y << "," << maxVtx.z << endl;
      std::cout << minVtx.x << "," << minVtx.y << "," << minVtx.z << endl;
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
   bool changed = true;
   int faceModel = 0;
   int paramMode = 0;
   bool pause = false;

   vector<vec3> vecVtx;

   sampleRN(
       mmMdFace.getExprNCoeffs(), mmFaceParams.bottomRows( mmMdFace.getExprNCoeffs() ).data() );
   sampleRN(
       mmMdFace.getShapeNCoeffs(), mmFaceParams.topRows( mmMdFace.getShapeNCoeffs() ).data() );

   sampleRN( BFaceMModel::NumCoeffs, bFaceParams.bottomRows( BFaceMModel::NumCoeffs ).data() );
   sampleRN( BFaceMModel::NumCoeffs, bFaceParams.topRows( BFaceMModel::NumCoeffs ).data() );

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
                  case SDLK_SPACE:
                     pause = !pause;
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
                     faceModel = faceModel ? 0 : 1;
                     changed = true;
                     SDL_ShowCursor( SDL_ENABLE );
                     break;
                  case SDL_BUTTON_RIGHT:
                     paramMode = paramMode ? 0 : 1;
                     changed = true;
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
               changed = true;
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
      glClear( GL_DEPTH_BUFFER_BIT );
      /*glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();
      draw( texA.id, texA.sz.x, texA.sz.y );
      drawFaces( imgFaces, imgFacesKps );
      glPopMatrix();*/

      static float ts = 85.0;
      static float tsInc = 1.0;
      if ( abs( ts ) >= 85.0 ) tsInc *= -1.0;

      if ( changed )
      {
         changed = false;

         if ( faceModel == 0 )
         {
            if ( paramMode == 0 )
            {
               sampleRN(
                   mmMdFace.getExprNCoeffs(),
                   mmFaceParams.bottomRows( mmMdFace.getExprNCoeffs() ).data() );
            }
            else
            {
               sampleRN(
                   mmMdFace.getShapeNCoeffs(),
                   mmFaceParams.topRows( mmMdFace.getShapeNCoeffs() ).data() );
            }

            mmMdFace.getVerticesFromNormParams( mmFaceParams, vecVtx );
            const std::vector<glm::uvec3>& vecIdx = mmMdFace.getFaces();
            mdFaceA.load(
                vecVtx.size(),
                vecVtx.empty() ? nullptr : &vecVtx[0],
                vecIdx.size(),
                vecIdx.empty() ? nullptr : &vecIdx[0] );
         }
         else
         {
            if ( paramMode == 0 )
            {
               sampleRN(
                   BFaceMModel::NumCoeffs, bFaceParams.topRows( BFaceMModel::NumCoeffs ).data() );
            }
            else
            {
               sampleRN(
                   BFaceMModel::NumCoeffs,
                   bFaceParams.bottomRows( BFaceMModel::NumCoeffs ).data() );
            }

            vecVtx.resize( BFaceMModel::NumVertices, vec3( 0.0f ) );
            std::vector<vec3> vtxCol( BFaceMModel::NumVertices, vec3( 0.0f ) );
            bFaceMMd.get(
                bFaceParams.data(),
                bFaceParams.data() + BFaceMModel::NumCoeffs,
                glm::value_ptr( vecVtx[0] ),
                glm::value_ptr( vtxCol[0] ) );

            mdFaceA.load(
                vecVtx.size(),
                vecVtx.empty() ? nullptr : &vecVtx[0],
                BFaceMModel::NumFaces,
                bFaceMMd.getFaces() );
            mdFaceA.loadAttrib( 3, value_ptr( vtxCol[0] ) );
         }
      }

      drawFaceModel( mdFaceA, ts );

      if (faceModel==1)
      {
         const auto colour = vec4( 0.0, 1.0, 0.0, 0.85 );
         glColor4f( 0.0, 0.1, 0.85, 1.0 );
         glMatrixMode( GL_MODELVIEW );
         glPushMatrix();
         glTranslatef( 0.5 * windowSz.x, 0.5 * windowSz.y, 0.0 );
         glScalef( 2.0, 2.0, 0.0 );
         glRotatef( 180.0, 0.0, 0.0, 1.0 );
         glRotatef( ts, 0.0, 1.0, 0.0 );
         for ( const auto& fps : BFaceMModel::getLandmarksIdx() )
         {
            const float r = 3.0;
            auto pos = vecVtx[fps];
            glBegin( GL_TRIANGLE_FAN );
            glColor4fv( glm::value_ptr( colour * 0.5f ) );
            glVertex3fv( value_ptr( pos ) );
            glColor4fv( value_ptr( colour ) );
            for ( int n = 0; n <= 128; ++n )
            {
              float const t = 2 * M_PI * (float)n / (float)128;
              glVertex3f( pos.x + sin( t ) * r, pos.y + cos( t ) * r, pos.z );
            }
            glEnd();
         }
         glPopMatrix();
         glColor4f( 1.0, 1.0, 1.0, 1.0 );
      }

      if (!pause) ts += tsInc;

      SDL_GL_SwapWindow( window );
      if ( 1000 / 60 > ( SDL_GetTicks() - start ) )
         SDL_Delay( 1000 / 60 - ( SDL_GetTicks() - start ) );
   }
   SDL_Quit();

   return ( 0 );
}