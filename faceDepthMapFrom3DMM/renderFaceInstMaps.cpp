/*!
 * *****************************************************************************
 *   \file faceDepthMapFrom3D_main.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-03-19
 *   *****************************************************************************/

#include "utils/gl_utils.h"
#include "utils/gl_utils.inline.h"
#include "utils/imgFileLst.h"

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

#include <boost/filesystem.hpp>

#include <memory>
#include <random>
#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;
using namespace boost;

namespace
{
std::random_device rd{};
std::mt19937 rs_gen{rd()};

void init()
{
   glClearColor( 0.0, 0.0, 0.0, 0.0 );
   glMatrixMode( GL_PROJECTION );
   glLoadIdentity();
   // glOrtho( 0.0, windowSz.x, windowSz.y, 0.0, -10000.0, 10000.0 );
   // glEnable( GL_BLEND );
   glEnable( GL_DEPTH_TEST );
   // glFrontFace( GL_CW );
   glEnable( GL_TEXTURE_2D );
   glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
}

void drawFaceModel(
    const gl_utils::TriMeshBuffer& mdFace,
    const glm::mat4& p,
    const glm::mat4& mv,
    const glm::vec3& lightPos,
    const glm::vec3& lightCol,
    const float ambient )
{
   static gl_utils::RenderProgram faceShader;

   static GLint uniMVP = -1;
   static GLint uniMV = -1;
   static GLint uniMVN = -1;
   static GLint uniLightPos = -1;
   static GLint uniLightCol = -1;
   static GLint uniAmbient = -1;

   if ( faceShader._id == -1 )
   {
      faceShader.load(
          "/mnt/p4/avila/moennen_wkspce/sceneIllEst/faceDepthMapFrom3DMM/shaders/face_frag.glsl",
          "/mnt/p4/avila/moennen_wkspce/sceneIllEst/faceDepthMapFrom3DMM/shaders/face_vtx.glsl" );
      uniMVP = faceShader.getUniform( "mvp" );
      uniMV = faceShader.getUniform( "mv" );
      uniMVN = faceShader.getUniform( "mvn" );
      uniLightPos = faceShader.getUniform( "lightPos" );
      uniLightCol = faceShader.getUniform( "lightColor" );
      uniAmbient = faceShader.getUniform( "ambient" );
   }

   faceShader.activate();

   glUniformMatrix4fv( uniMV, 1, 0, value_ptr( mv ) );

   const mat4 mvp = p * mv;
   glUniformMatrix4fv( uniMVP, 1, 0, value_ptr( mvp ) );
   const mat4 mvt = transpose( inverse( mv ) );
   glUniformMatrix4fv( uniMVN, 1, 0, value_ptr( mvt ) );
   glUniform3fv( uniLightPos, 1, value_ptr( lightPos ) );
   glUniform3fv( uniLightCol, 1, value_ptr( lightCol ) );

   glUniform1f( uniAmbient, ambient );

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

void getOutUVS(
    const vector<vec2>& iuvs,
    const vector<uvec3>& ivecIdx,
    const glm::uvec3* ovecIdx,
    glm::vec2* ouvs )
{
   for ( size_t f = 0; f < ivecIdx.size(); ++f )
   {
      ouvs[ovecIdx[f].x] = iuvs[ivecIdx[f].x];
      ouvs[ovecIdx[f].y] = iuvs[ivecIdx[f].y];
      ouvs[ovecIdx[f].z] = iuvs[ivecIdx[f].z];
   }
}
}
const string keys =
    "{help h usage ? |         | print this message   }"
    "{@faceModel     |         | face detection model }"
    "{@faceInstLst   |         | face model instance files list}"
    "{@outDir        |         | output directories   }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }

   // Load background images
   ImgNFileLst instLst(2, parser.get<string>( "@faceInstLst" ).c_str(), "" );
   if ( instLst.size() == 0 )
   {
      cerr << "Invalid face instances list : " << parser.get<string>( "@faceInstLst" ) << endl;
      return -1;
   }

   //------------------------------------------------------------------------- Load model
   //
   const string BEfaceModelBin = parser.get<string>( "@faceModel" );
   BEFaceMModel beFaceMMd( BEfaceModelBin.c_str() );
   if ( !beFaceMMd.initialized() )
   {
      std::cerr << "Cannot import model : " << BEfaceModelBin << std::endl;
      return -1;
   }

   uvec2 imgSz( 256, 256 );

   // SDL init
   SDL_Init( SDL_INIT_EVERYTHING );
   SDL_Window* window = SDL_CreateWindow(
       "renderFaceMaps",
       SDL_WINDOWPOS_CENTERED,
       SDL_WINDOWPOS_CENTERED,
       imgSz.x,
       imgSz.y,
       SDL_WINDOW_OPENGL );
   SDL_GLContext glCtx = SDL_GL_CreateContext( window );

   // GLEW init
   GLenum err = glewInit();
   if ( GLEW_OK != err )
   {
      cerr << "Error: " << glewGetErrorString( err ) << endl;
      return -1;
   }

   // Common data alloc
   vector<vec3> vecVtx( BEFaceMModel::NumVertices, vec3( 0.0f ) );
   vector<vec3> vtxCol( BEFaceMModel::NumVertices, vec3( 0.0f ) );
   vector<vec3> vtxNorm( BEFaceMModel::NumVertices, vec3( 0.0f ) );

   // Out root path
   const filesystem::path outRootPath( parser.get<string>( "@outDir" ) );

   // Constant render params
   const float ambient(1.0);
   const vec3 lightPos(0.0);
   const vec3 lightCol = vec3(0.1);

   for ( int s = 0; s < instLst.size(); ++s )
   {
      Mat instImg = cv_utils::imread32FC4( instLst.filePath(s,0) );
      if ( instImg.empty() )
      {
         std::cerr << "Cannot load instance image : " << instLst.filePath(s,0) << std::endl;
         continue;
      }
      imgSz = uvec2( instImg.cols, instImg.rows );

      BEFaceMModelInstance instModel = { instLst.filePath(s,1) };
      if ( !instModel.initialized() )
      {
         std::cerr << "Cannot load instance model : " <<  instLst.filePath(s,1) << std::endl;
         continue;
      }

      // Set the render buffers
      gl_utils::RenderTarget renderTarget( imgSz );
      gl_utils::Texture<gl_utils::RGBA_32FP> faceColorTex( imgSz );
      gl_utils::Texture<gl_utils::RGBA_32FP> faceUVDepthTex( imgSz );
      gl_utils::Texture<gl_utils::RGBA_32FP> faceNormalsTex( imgSz );
      GLuint rTex[3] = {faceColorTex.id, faceUVDepthTex.id, faceNormalsTex.id};

      renderTarget.bind( 3, &rTex[0] );
      glClear( GL_COLOR_BUFFER_BIT );
      glClear( GL_DEPTH_BUFFER_BIT );
      renderTarget.unbind();

      // Load background
      gl_utils::uploadToTexture( faceColorTex, instImg.ptr() );

      // Create the face geometry
      beFaceMMd.get(
          instModel.getShapeCoeffs(),
          instModel.getExpCoeffs(),
          instModel.getTexCoeffs(),
          glm::value_ptr( vecVtx[0] ),
          glm::value_ptr( vtxCol[0] ),
          false );

      gl_utils::TriMeshBuffer mdFaceA;

      // vertices
      mdFaceA.load(
          vecVtx.size(),
          vecVtx.empty() ? nullptr : &vecVtx[0],
          BEFaceMModel::NumFaces,
          beFaceMMd.getFaces() );

      // color
      mdFaceA.loadAttrib( 3, value_ptr( vtxCol[0] ) );

      // normals
      gl_utils::computeNormals(
          BEFaceMModel::NumFaces,
          beFaceMMd.getFaces(),
          BEFaceMModel::NumVertices,
          &vecVtx[0],
          &vtxNorm[0] );
      mdFaceA.loadAttrib( 3, value_ptr( vtxNorm[0] ) );

      // uvs
      mdFaceA.loadAttrib( 2, value_ptr( beFaceMMd.getUVs()[0] ) );

      //  Camera projection
      const mat4 camProj = glm::ortho(0.0f, static_cast<float>(imgSz.x), 
        static_cast<float>(imgSz.y), 0.0f, -10000.0f, 10000.0f );
             
      // Model view
      const mat4 modelView;// = instModel.getPose();

      // Draw the face
      renderTarget.bind( 3, &rTex[0] );

      glEnable( GL_DEPTH_TEST );
      glEnable( GL_CULL_FACE );
      glCullFace( GL_BACK );

      drawFaceModel( mdFaceA, camProj, modelView, lightPos, lightCol, ambient );

      renderTarget.unbind();

      // upload and write the maps
      char sampleId[16];
      sprintf( sampleId, "%08d_", s );
      const string outBasename( sampleId );
      const string outBasenameFull = ( outRootPath / filesystem::path( sampleId ) ).string();

      Mat faceColorImg( faceColorTex.sz.y, faceColorTex.sz.x, CV_32FC4 );
      Mat faceUVDepthImg( faceUVDepthTex.sz.y, faceUVDepthTex.sz.x, CV_32FC4 );
      Mat faceNormalsImg( faceNormalsTex.sz.y, faceNormalsTex.sz.x, CV_32FC4 );
      if ( gl_utils::readbackTexture( faceColorTex, faceColorImg.data ) &&
           gl_utils::readbackTexture( faceUVDepthTex, faceUVDepthImg.data ) &&
           gl_utils::readbackTexture( faceNormalsTex, faceNormalsImg.data ) )
      {
         cvtColor( faceColorImg, faceColorImg, cv::COLOR_RGBA2BGR );
         cvtColor( faceUVDepthImg, faceUVDepthImg, cv::COLOR_RGBA2BGR );
         cvtColor( faceNormalsImg, faceNormalsImg, cv::COLOR_RGBA2BGR );

         imwrite( outBasenameFull + "c.png", faceColorImg * 255.0 );
         imwrite( outBasenameFull + "uvd.exr", faceUVDepthImg );
         imwrite( outBasenameFull + "n.exr", faceNormalsImg );

         std::cout << outBasename + "c.png " << outBasename + "uvd.exr " << outBasename + "n.exr"
                   << std::endl;

         imshow( "color", faceColorImg );
         imshow( "uvdepth", faceUVDepthImg );
         imshow( "normals", faceNormalsImg );
         waitKey( 0 );
      }
   }

   SDL_Quit();

   return ( 0 );
}