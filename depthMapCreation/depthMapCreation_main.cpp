/*!
* *****************************************************************************
*   \file depthMapCreation_main.cpp
*   \author moennen
*   \brief
*   \date 2018-02-19
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

#include <memory>

#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@imageA        |         | image1 for compare   }";

static ivec2 windowSz( 1024, 768 );

#define INPUTA_MODE 0
#define MAP_MODE 2

using MapFunc = PhSpline<2, 3, 2>;
using MapFuncPtr = unique_ptr<MapFunc>;

MapFuncPtr createNewMapFunc( const vector<vec2>& mapCtrlPtx, const Mat& input )
{
   vector<vec2> mapCtrlPos;
   mapCtrlPos.reserve( mapCtrlPtx.size() );
   vector<vec3> mapCtrlValues;
   mapCtrlValues.reserve( mapCtrlPtx.size() );
#pragma omp parallel for
   for ( size_t idx = 0; idx < mapCtrlPtx.size(); ++idx )
   {
      const auto ctrVal = cv_utils::imsample32FC3( input, mapCtrlPtx[idx] );
      mapCtrlPos.emplace_back(
          vec2( 2.0f ) * ( mapCtrlPtx[idx] / vec2( windowSz ) - vec2( 0.5f ) ) );
      mapCtrlValues.emplace_back( ctrVal( 0 ), ctrVal( 1 ), ctrVal( 2 ) );
   }

   return MapFuncPtr( new MapFunc(
       value_ptr( mapCtrlPos[0] ), value_ptr( mapCtrlValues[0] ), mapCtrlPtx.size() ) );
}

void computeMap( const MapFunc& mapFunc, const vector<vec2>& mapCtrlPtx, Mat& map )
{
   vector<vec2> mapCtrlPos;
   mapCtrlPos.reserve( mapCtrlPtx.size() );
   for ( size_t idx = 0; idx < mapCtrlPtx.size(); ++idx )
   {
      mapCtrlPos.emplace_back(
          vec2( 2.0f ) * ( mapCtrlPtx[idx] / vec2( windowSz ) - vec2( 0.5f ) ) );
   }

   const auto ctrlPtx = value_ptr( mapCtrlPos[0] );
#pragma omp parallel for
   for ( size_t y = 0; y < map.rows; y++ )
   {
      float* row_data = map.ptr<float>( y );
      vec3 val;
      for ( size_t x = 0; x < map.cols; x++ )
      {
         mapFunc(
             ctrlPtx,
             value_ptr( vec2( 2.0f ) * ( vec2( x, y ) / vec2( windowSz ) - vec2( 0.5f ) ) ),
             value_ptr( val ) );
         row_data[x * 3 + 0] = val.x;
         row_data[x * 3 + 1] = val.y;
         row_data[x * 3 + 2] = val.z;
      }
   }
}

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

size_t getSelectedPt( const vec2 pos, const vector<vec2>& ptx )
{
   const float defaultSelRadius = 7.0;
   for ( size_t i = 0; i < ptx.size(); ++i )
   {
      const vec2 pt = {ptx[i].x, ptx[i].y};
      if ( distance( pt, pos ) <= defaultSelRadius ) return i;
   }
   return ptx.size();
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

void drawCtrlPtx( const vector<vec2>& ctrlPtx, const size_t sel )
{
   const float defaultRadius = 5.0;

   for ( size_t i = 0; i < ctrlPtx.size(); ++i )
   {
      const vec2& pt = ctrlPtx[i];
      const float alpha = 0.85;
      const vec4 colour =
          ( i == sel ) ? vec4( 0.75f, 0.0f, 0.0f, alpha ) : vec4( 0.0f, 0.75f, 0.0f, alpha );

      drawCircle( pt.x, pt.y, defaultRadius, 128, colour );
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

   /*auto func = [](const vec2 x) {
     return exp(0.5 * x.x * x.x) * exp(0.5 * x.y * x.y);
   };
   vector<vec2> ctrlX = {
       {0.3f, -0.1f}, {-0.04f, 0.8f}, {-0.97f, 0.001f}, {0.75f, -1.3f}};
   vector<float> ctrlY;
   ctrlY.reserve(ctrlX.size());
   transform(ctrlX.cbegin(), ctrlX.cend(), std::back_inserter(ctrlY), func);

   PhSpline<2, 1, 2, float> interpFunc(value_ptr(ctrlX[0]), &ctrlY[0],
                                       ctrlX.size());

   for (float x = -2.0f; x <= 2.0f; x += 0.01f) {
     // for (float y = -1.0f; y <= 1.0f; y += 0.1f) {
     const vec2 xy(x, 0.0f);
     const float rf = func(xy);
     float ef;
     interpFunc(value_ptr(ctrlX[0]), value_ptr(xy), &ef);
     cout << x << " " << rf << " " << ef << endl;
   }
   //}*/

   /*auto func = []( const float x ) { return x < 0.0f ? 0.0f : exp( 0.5 * x * x ); };
   vector<float> ctrlX = {
       -2.0f, 1.0f, 1.5f, 0.3f, -0.1f, 0.1f, 0.5f, -0.6f, -0.04f, 0.8f, -1.75f, -1.3f, 2.0f};
   vector<float> ctrlY;
   ctrlY.reserve( ctrlX.size() );
   transform( ctrlX.cbegin(), ctrlX.cend(), std::back_inserter( ctrlY ), func );

   PhSpline<1, 1, 2, float> interpFunc( &ctrlX[0], &ctrlY[0], ctrlX.size() );

   for ( float x = -2.0f; x <= 2.0f; x += 0.01f )
   {
      const float rf = func( x );
      float ef;
      interpFunc( &ctrlX[0], &x, &ef );
      cout << x << " " << rf << " " << ef << endl;
   }*/

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
       "shLighting",
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

   // create the map model
   MapFuncPtr mapFuncPtr;

   bool running = true;
   int displayMode = INPUTA_MODE;
   bool updateMap = false;
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
                     displayMode = INPUTA_MODE;
                     break;
                  case SDLK_2:
                     displayMode = MAP_MODE;
                     break;
               }
            case SDL_MOUSEBUTTONDOWN:
               switch ( event.button.button )
               {
                  case SDL_BUTTON_LEFT:
                  {
                     selectedDepthCtrlPtx =
                         getSelectedPt( glm::vec2( event.motion.x, event.motion.y ), depthCtrlPtx );
                     const bool hasSelection = selectedDepthCtrlPtx < depthCtrlPtx.size();
                     // no selection : create a new ctrl pt
                     if ( !hasSelection )
                     {
                        // add
                        depthCtrlPtx.emplace_back( event.motion.x, event.motion.y );
                     }
                     updateMap = true;
                     // disable mouse cursor
                     SDL_ShowCursor( SDL_DISABLE );
                     break;
                  }
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
                     // release the current selection
                     selectedDepthCtrlPtx = depthCtrlPtx.size();
                     // disable mouse cursor
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
               int mouseX = event.motion.x;
               int mouseY = event.motion.y;
               std::stringstream ss;
               ss << "X: " << mouseX << " Y: " << mouseY;
               SDL_SetWindowTitle( window, ss.str().c_str() );
               {
                  const bool hasSelection = selectedDepthCtrlPtx < depthCtrlPtx.size();
                  // no selection : create a new ctrl pt
                  if ( hasSelection )
                  {
                     depthCtrlPtx[selectedDepthCtrlPtx].x = mouseX;
                     depthCtrlPtx[selectedDepthCtrlPtx].y = mouseY;
                     updateMap = true;
                  }
               }
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
      if ( updateMap && ( displayMode == MAP_MODE ) )
      {
         mapFuncPtr = createNewMapFunc( depthCtrlPtx, inputA );
         computeMap( *mapFuncPtr.get(), depthCtrlPtx, map );
         gl_utils::uploadToTexture( texMap, map.ptr() );
         updateMap = false;
      }
      glClear( GL_COLOR_BUFFER_BIT );
      if ( displayMode == INPUTA_MODE )
      {
         draw( texA.id, texA.sz.x, texA.sz.y );
      }
      else
      {
         draw( texMap.id, texMap.sz.x, texMap.sz.y );
      }
      // updateDepthTextureFromCtrlPtx( depthTex, depthCtrlPtx );
      // draw( depthTex.id, depthTex.sz.x, depthTex.sz.y );
      drawCtrlPtx( depthCtrlPtx, selectedDepthCtrlPtx );
      SDL_GL_SwapWindow( window );
      if ( 1000 / 60 > ( SDL_GetTicks() - start ) )
         SDL_Delay( 1000 / 60 - ( SDL_GetTicks() - start ) );
   }
   SDL_Quit();

   return ( 0 );
}
