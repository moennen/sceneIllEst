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

#include <boost/filesystem.hpp>
#include <Eigen/Dense>

#include <memory>
#include <random>
#include <iostream>

#include "skinColourSamples.h"

using namespace std;
using namespace cv;
using namespace glm;
using namespace boost;

namespace
{
std::random_device rd{};
std::mt19937 rs_gen{rd()};

template <typename T>
void sampleRN( const size_t d, T* data )
{
   std::normal_distribution<T> rd;

   for ( size_t i = 0; i < d; ++i ) data[i] = rd( rs_gen );
}

vec3 sampleSkinColourFromData()
{
   static uniform_int_distribution<> rs_sample( 0, 50859 );

   const int sample = rs_sample( rs_gen );

   return vec3(
       skinColourSamples[3 * sample + 2] / 255.0,
       skinColourSamples[3 * sample + 1] / 255.0,
       skinColourSamples[3 * sample] / 255.0 );
}

vec3 sampleSkinColourFromModel()
{
   static bool isInit = false;
   static Eigen::MatrixXd skinRGB( 3, 3 );
   static Eigen::VectorXd skinColMean( 3 );
   static normal_distribution<> rs_skin_color( 0.0, 1.5 );

   // create the transform matrix from random number to skin rgb values
   if ( !isInit )
   {
      // learn from Skin_NonSkin.txt
      skinColMean << 203.992, 146.601, 113.87;
      Eigen::MatrixXd skinColCovar( 3, 3 );
      skinColCovar.row( 0 ) << 1421.03, 1296.27, 1311.81;
      skinColCovar.row( 1 ) << 1296.27, 1284.54, 1379.75;
      skinColCovar.row( 2 ) << 1311.81, 1379.75, 1731.63;

      Eigen::LLT<Eigen::MatrixXd> cholSolver( skinColCovar );
      skinRGB = cholSolver.matrixL();

      isInit = true;
   }

   Eigen::VectorXd val( 3 );
   val << rs_skin_color( rs_gen ), rs_skin_color( rs_gen ), rs_skin_color( rs_gen );

   val = skinRGB * val + skinColMean;

   return vec3( val[0] / 255.0, val[1] / 255.0, val[2] / 255.0 );
}

vec3 sampleSkinColourOffset( const vector<vec3>& vtxCol )
{
   static uniform_real_distribution<float> rs_method( 0, 1.0 );
   static uniform_real_distribution<float> rs_gain( 0.1, 1.0 );

   const float methodRand = rs_method( rs_gen );

   vec3 skinColour( 1.0 );

   if ( methodRand < 0.2 ) return skinColour;

   skinColour = ( methodRand < 0.3 ) ? sampleSkinColourFromModel() : sampleSkinColourFromData();

   vec3 vtxColour( 0.0 );
   for ( auto col : vtxCol )
   {
      vtxColour += col;
   }
   vtxColour /= vtxCol.size();

   return rs_gain( rs_gen ) * skinColour / vtxColour;
}

void fittSz( Mat& img, const uvec2 sampleSz, const float scale, const float tx, const float ty )
{
   constexpr float maxDsScaleFactor = 3.5;

   uvec2 imgSz( img.cols, img.rows );
   // random rescale
   const float minDs = std::max( (float)sampleSz.y / imgSz.y, (float)sampleSz.x / imgSz.x );
   const float ds = mix( std::min( 1.0f, maxDsScaleFactor * minDs ), minDs, scale );
   resize( img, img, Size(), ds, ds, CV_INTER_AREA );
   imgSz = ivec2( img.cols, img.rows );

   // random translate
   const ivec2 trans(
       std::floor( tx * ( imgSz.x - sampleSz.x ) ), std::floor( ty * ( imgSz.y - sampleSz.y ) ) );

   // crop
   img = img( Rect( trans.x, trans.y, sampleSz.x, sampleSz.y ) ).clone();
}

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
    const vec3& faceColOffset,
    const mat4& p,
    const mat4& mv,
    const vec3& lightPos,
    const vec3& lightCol,
    const float ambient )
{
   static gl_utils::RenderProgram faceShader;

   static GLint uniMVP = -1;
   static GLint uniMV = -1;
   static GLint uniMVN = -1;
   static GLint uniPosFaceCenter = -1;
   static GLint uniLightPos = -1;
   static GLint uniLightCol = -1;
   static GLint uniAmbient = -1;
   static GLint uniColOffset = -1;
   static GLint uniRoughness = -1;
   static GLint uniSubsurface = -1;
   static GLint uniSheen = -1;

   static normal_distribution<> rs_skin_roughness( 0.0, 0.1 );
   static normal_distribution<> rs_skin_subsurface( 0.0, 0.1 );
   static normal_distribution<> rs_skin_sheen( 0.5, 0.1 );

   if ( faceShader._id == -1 )
   {
      faceShader.load(
          "/mnt/p4/avila/moennen_wkspce/sceneIllEst/faceDepthMapFrom3DMM/shaders/face_frag.glsl",
          "/mnt/p4/avila/moennen_wkspce/sceneIllEst/faceDepthMapFrom3DMM/shaders/face_vtx.glsl" );
      uniMVP = faceShader.getUniform( "mvp" );
      uniMV = faceShader.getUniform( "mv" );
      uniMVN = faceShader.getUniform( "mvn" );
      uniPosFaceCenter = faceShader.getUniform( "posFaceCenter" );
      uniLightPos = faceShader.getUniform( "lightPos" );
      uniLightCol = faceShader.getUniform( "lightColor" );
      uniAmbient = faceShader.getUniform( "ambient" );
      uniColOffset = faceShader.getUniform( "colourOffset" );
      uniRoughness = faceShader.getUniform( "roughness" );
      uniSubsurface = faceShader.getUniform( "subsurface" );
      uniSheen = faceShader.getUniform( "sheen" );
   }

   faceShader.activate();

   glUniformMatrix4fv( uniMV, 1, 0, value_ptr( mv ) );

   const mat4 mvp = p * mv;
   glUniformMatrix4fv( uniMVP, 1, 0, value_ptr( mvp ) );
   const mat4 mvt = transpose( inverse( mv ) );
   glUniformMatrix4fv( uniMVN, 1, 0, value_ptr( mvt ) );

   const vec4 v4PosFaceCenter = mvp * vec4( 0.0, 0.0, 0.0, 1.0 );
   const vec2 posFaceCenter(
       v4PosFaceCenter.x / v4PosFaceCenter.w, v4PosFaceCenter.y / v4PosFaceCenter.w );
   glUniform2fv( uniPosFaceCenter, 1, value_ptr( posFaceCenter ) );

   glUniform3fv( uniLightPos, 1, value_ptr( lightPos ) );
   glUniform3fv( uniLightCol, 1, value_ptr( lightCol ) );

   glUniform1f( uniAmbient, ambient );

   glUniform1f( uniRoughness, clamp( 1.0 - abs( rs_skin_roughness( rs_gen ) ), 0.0, 1.0 ) );
   glUniform1f( uniSubsurface, clamp( 1.0 - abs( rs_skin_subsurface( rs_gen ) ), 0.0, 1.0 ) );
   glUniform1f( uniSheen, clamp( rs_skin_sheen( rs_gen ), 0.0, 1.0 ) );

   glUniform3fv( uniColOffset, 1, value_ptr( faceColOffset ) );

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
    "{@backImgLst    |         | background image list}"
    "{@nRenders      |         | number of renders    }"
    "{@startIdx      |         | idx of the first render }"
    "{@outDir        |         | output directories   }"
    "{show           |         |    }"
    "{nowrite        |        |    }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }

   const bool doShow = parser.get<bool>( "show" );
   const bool doWrite = !parser.get<bool>( "nowrite" );

   // Load background images
   ImgNFileLst imgLst( 1, parser.get<string>( "@backImgLst" ).c_str(), "" );
   if ( imgLst.size() == 0 )
   {
      cerr << "Invalid background image list : " << parser.get<string>( "@backImgLst" ) << endl;
      return -1;
   }
   uniform_int_distribution<> rs_img( 0, imgLst.size() - 1 );

   //------------------------------------------------------------------------- Load model
   //
   string BEfaceModelBin = parser.get<string>( "@faceModel" );

   BEFaceMModel beFaceMMd( BEfaceModelBin.c_str() );
   const unsigned nFaceParamsCoeffs = 2 * BEFaceMModel::NumCoeffs + BEFaceMModel::NumExpCoeffs;
   Eigen::VectorXf beFaceParams = Eigen::VectorXf::Zero( nFaceParamsCoeffs );

   if ( !beFaceMMd.initialized() )
   {
      vector<uvec3> vecIdx;
      vector<vec3> vecVtx;
      vector<vec2> vecUvs;
      vector<vec3> vecNormals;
      vector<vec4> vecColors;
      std::cerr << "Cannot import model -> no uvs ? " << BEfaceModelBin << std::endl;
      if ( gl_utils::loadTriangleMesh(
               "/tmp/beFaceMd.fbx", vecIdx, vecVtx, vecUvs, vecNormals, vecColors ) &&
           ( vecIdx.size() >= BEFaceMModel::NumFaces ) )
      {
         vector<vec2> ovecUvs( BEFaceMModel::NumVertices );
         getOutUVS( vecUvs, vecIdx, beFaceMMd.getFaces(), &ovecUvs[0] );
         beFaceMMd = BEFaceMModel( BEfaceModelBin.c_str(), &ovecUvs[0] );
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

   const uvec2 imgSz( 320, 320 );

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

   // Sampler
   normal_distribution<> rs_fov( 65.0, 11.5 );

   normal_distribution<> rs_yaw( 0.0, 21.0 );
   normal_distribution<> rs_pitch( 0.0, 10.5 );
   normal_distribution<> rs_roll( 0.0, 7.0 );

   uniform_real_distribution<> rs_pos_xy( 0.25, 0.75 );
   uniform_real_distribution<> rs_pos_z( -800.0, -550.0 );
   normal_distribution<> rs_scale_off( 0.0, 1.325 );

   uniform_real_distribution<> rs_shade( 0.05, 1.0 );
   uniform_real_distribution<> rs_lightPos( -500.0, 500.0 );
   uniform_real_distribution<> rs_lightCol( 0.5, 15.0 );
   uniform_real_distribution<> rs_ambient( 0.5, 7.0 );

   uniform_real_distribution<> rs_backImgResize( 0.0, 1.0 );

   // Out root path
   const filesystem::path outRootPath( parser.get<string>( "@outDir" ) );

   const int nRenders = parser.get<int>( "@nRenders" );
   const int startIdx = parser.get<int>( "@startIdx" );
   normal_distribution<> rs_nfaces( 0.0, 0.915 );

   const int nMaxRendersPerGroup = 10000;
   unsigned renderGroupId = startIdx / nMaxRendersPerGroup;

   // Set the render buffers
   gl_utils::RenderTarget renderTarget( imgSz );
   gl_utils::Texture<gl_utils::RGBA_32FP> faceColorTex( imgSz );
   gl_utils::Texture<gl_utils::RGBA_32FP> faceUVDepthTex( imgSz );
   gl_utils::Texture<gl_utils::RGBA_32FP> faceNormalsTex( imgSz );
   gl_utils::Texture<gl_utils::RGBA_32FP> faceIdMatteTex( imgSz );
   GLuint rTex[4] = {faceColorTex.id, faceUVDepthTex.id, faceNormalsTex.id, faceIdMatteTex.id};

   for ( int s = 0; s < nRenders; ++s )
   {
      if ( ( ( startIdx + s ) % nMaxRendersPerGroup ) == 0 )
      {
         renderGroupId += 1;
         char dirname[7];
         sprintf( dirname, "%06d", renderGroupId );
         const filesystem::path outGroupRootPath =
             outRootPath / filesystem::path( std::string( dirname ) );
         if ( !filesystem::create_directory( outGroupRootPath ) )
         {
            cerr << "Cannot create directory : " << outGroupRootPath.string() << endl;
         }
      }

      // Sample a random background images
      const size_t si = rs_img( rs_gen );
      Mat backImg = cv_utils::imread32FC4( imgLst.filePath( si, 0 ) );
      if ( ( backImg.cols < imgSz.x ) || ( backImg.rows < imgSz.y ) )
      {
         --s;
         continue;
      }

      fittSz(
          backImg,
          imgSz,
          rs_backImgResize( rs_gen ),
          rs_backImgResize( rs_gen ),
          rs_backImgResize( rs_gen ) );

      renderTarget.bind( 4, &rTex[0] );
      glClear( GL_COLOR_BUFFER_BIT );
      glClear( GL_DEPTH_BUFFER_BIT );
      renderTarget.unbind();

      // Load background
      gl_utils::uploadToTexture( faceColorTex, backImg.ptr() );

      // Sample the prespective view
      const float fov = clamp( rs_fov( rs_gen ), 15.0, 110.0 );
      const mat4 camProj = glm::perspectiveFov(
          (float)( fov * M_PI / 180.0 ), (float)imgSz.x, (float)imgSz.y, 0.0001f, 10000.0f );
      const vec4 camProjectionInfo = vec4(
          -2.0 / ( imgSz.x * camProj[0][0] ),
          -2.0 / ( imgSz.y * camProj[1][1] ),
          ( 1.0 - camProj[0][2] ) / camProj[0][0],
          ( 1.0 + camProj[1][2] ) / camProj[1][1] );

      // Sample the light
      const float shade = rs_shade( rs_gen );
      const float ambient = rs_ambient( rs_gen ) * ( 1.0 - shade );
      const vec3 lightPos =
          vec3( rs_lightPos( rs_gen ), rs_lightPos( rs_gen ), rs_lightPos( rs_gen ) );
      const vec3 lightCol = vec3( rs_lightCol( rs_gen ) ) * shade;

      // Sample a random number of faces
      const size_t nfaces = 1u + std::max( -1, static_cast<int>( rs_nfaces( rs_gen ) ) );

      const float sf = std::min( 1.0 + abs( rs_scale_off( rs_gen ) ), 3.5 );
      const float gz = rs_pos_z( rs_gen );

      for ( size_t f = 0; f < nfaces; ++f )
      {
         // Sample a random face
         sampleRN( nFaceParamsCoeffs, beFaceParams.data() );

         // Create the face geometry
         beFaceMMd.get(
             beFaceParams.data(),
             beFaceParams.data() + BEFaceMModel::NumCoeffs,
             beFaceParams.data() + BEFaceMModel::NumCoeffs + BEFaceMModel::NumExpCoeffs,
             glm::value_ptr( vecVtx[0] ),
             glm::value_ptr( vtxCol[0] ) );

         gl_utils::TriMeshBuffer mdFaceA;

         // vertices
         mdFaceA.load(
             vecVtx.size(),
             vecVtx.empty() ? nullptr : &vecVtx[0],
             BEFaceMModel::NumFaces,
             beFaceMMd.getFaces() );

         // color
         mdFaceA.loadAttrib( 3, value_ptr( vtxCol[0] ) );

         // sample a random skin colour offset
         const vec3 faceColOffset = sampleSkinColourOffset( vtxCol );

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

         // Sample the model view
         mat4 modelView;

         const float z = gz + 200.0 * f;
         const vec2 ss_pos = vec2( rs_pos_xy( rs_gen ) * imgSz.x, rs_pos_xy( rs_gen ) * imgSz.y );
         const vec3 cs_pos = vec3(
             ( ss_pos * vec2( camProjectionInfo.x, camProjectionInfo.y ) +
               vec2( camProjectionInfo.z, camProjectionInfo.w ) ) *
                 z,
             z );
         modelView = translate( modelView, cs_pos );
         const float pitch = clamp( rs_pitch( rs_gen ), -45.0, 45.0 );
         modelView = rotate( modelView, (float)( pitch * M_PI / 180.0 ), vec3( 1.0, 0.0, 0.0 ) );
         const float yaw = clamp( rs_yaw( rs_gen ), -90.0, 90.0 );
         modelView = rotate( modelView, (float)( yaw * M_PI / 180.0 ), vec3( 0.0, 1.0, 0.0 ) );
         const float roll = clamp( rs_roll( rs_gen ), -45.0, 45.0 );
         modelView =
             rotate( modelView, (float)( M_PI + roll * M_PI / 180.0 ), vec3( 0.0, 0.0, 1.0 ) );
         modelView = glm::scale( modelView, vec3( sf ) );

         // Draw the face
         renderTarget.bind( 4, &rTex[0] );

         glEnable( GL_DEPTH_TEST );
         glEnable( GL_CULL_FACE );
         glCullFace( GL_BACK );

         drawFaceModel( mdFaceA, faceColOffset, camProj, modelView, lightPos, lightCol, ambient );

         renderTarget.unbind();
      }

      // upload and write the maps
      char sampleId[16];
      sprintf( sampleId, "%06d/%08d_", renderGroupId, startIdx + s );
      const string outBasename( sampleId );
      const string outBasenameFull = ( outRootPath / filesystem::path( sampleId ) ).string();

      Mat faceColorImg( faceColorTex.sz.y, faceColorTex.sz.x, CV_32FC4 );
      Mat faceUVDepthImg( faceUVDepthTex.sz.y, faceUVDepthTex.sz.x, CV_32FC4 );
      Mat faceNormalsImg( faceNormalsTex.sz.y, faceNormalsTex.sz.x, CV_32FC4 );
      Mat faceIdMatteImg( faceIdMatteTex.sz.y, faceIdMatteTex.sz.x, CV_32FC4 );
      if ( gl_utils::readbackTexture( faceColorTex, faceColorImg.data ) &&
           gl_utils::readbackTexture( faceUVDepthTex, faceUVDepthImg.data ) &&
           gl_utils::readbackTexture( faceNormalsTex, faceNormalsImg.data ) &&
           gl_utils::readbackTexture( faceIdMatteTex, faceIdMatteImg.data ) )
      {
         cvtColor( faceColorImg, faceColorImg, cv::COLOR_RGBA2BGR );
         cvtColor( faceUVDepthImg, faceUVDepthImg, cv::COLOR_RGBA2BGR );
         cvtColor( faceNormalsImg, faceNormalsImg, cv::COLOR_RGBA2BGR );
         cvtColor( faceIdMatteImg, faceIdMatteImg, cv::COLOR_RGBA2BGR );

         if ( doWrite )
         {
            imwrite( outBasenameFull + "c.png", faceColorImg * 255.0 );
            imwrite( outBasenameFull + "uvd.exr", faceUVDepthImg );
            imwrite( outBasenameFull + "n.exr", faceNormalsImg );
            imwrite( outBasenameFull + "idm.exr", faceIdMatteImg );
         }

         std::cout << outBasename + "c.png " << outBasename + "uvd.exr " << outBasename + "n.exr "
                   << outBasename + "idm.exr" << std::endl;

         if ( doShow )
         {
            imshow( "color", faceColorImg );
            imshow( "uvdepth", faceUVDepthImg );
            imshow( "normals", faceNormalsImg );
            imshow( "idmatte", faceIdMatteImg );
            waitKey( 0 );
         }
      }
   }

   SDL_Quit();

   return ( 0 );
}