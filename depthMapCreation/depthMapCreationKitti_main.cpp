/*!
 * *****************************************************************************
 *   \file depthMapCreationHSBS_main.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-02-19
 *   *****************************************************************************/

#include "utils/imgFileLst.h"
#include "utils/cv_utils.h"

#include <Eigen/Dense>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <boost/filesystem.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;
using namespace boost;
using namespace Eigen;

namespace
{
//------------------------------------------------------------------------------
//
void processImg( Mat& img )
{
   img = img( Rect( 150, img.rows / 3 + 20, img.cols - 300, 2 * img.rows / 3 - 20 ) ).clone();
}

//------------------------------------------------------------------------------
//
void processDepth( Mat& depth, const Mat& img, const float undef,
                  const float sp_z, const float col_z, const float filter )
{
   depth =
       depth( Rect( 150, depth.rows / 3 + 20, depth.cols - 300, 2 * depth.rows / 3 - 20 ) ).clone() ;

   vector<Mat> dPyr;
   dPyr.emplace_back( depth.rows, depth.cols, CV_32FC4 );

   // fill the first level
#pragma omp parallel for
   for ( unsigned y = 0; y < depth.rows; y++ )
   {
      const float* depthPtr = depth.ptr<float>( y );
      vec4* dPtr = dPyr.back().ptr<vec4>( y );
      for ( unsigned x = 0; x < depth.cols; x++ )
      {
         const float d = depthPtr[x];
         dPtr[x] = d == undef ? vec4( 0.0 )
                              : vec4( d, 1.0f, static_cast<float>( x ), static_cast<float>( y ) );
      }
   }

   vector<Mat> iPyr;
   iPyr.push_back( img.clone() );

   unsigned currPyrSz = std::min( depth.rows, depth.cols );

   // downscale : integrate the depth value using position
   while ( currPyrSz >= 2 )
   {
      Mat iCurr;
      resize( iPyr.back(), iCurr, Size( 0, 0 ), 0.5, 0.5, INTER_AREA );
      Mat dCurr( iCurr.rows, iCurr.cols, CV_32FC4 );

#pragma omp parallel for
      for ( unsigned y = 0; y < dCurr.rows; y++ )
      {
         vec4* depthPtr = dCurr.ptr<vec4>( y );

         for ( unsigned x = 0; x < dCurr.cols; x++ )
         {
            vec4 val( 0.0 );
            float w = 0.0;

            const vec2 pos( x * 2.f, y * 2.f );

            for ( int dy = -1; dy <= 1; dy++ )
            {
               for ( int dx = -1; dx <= 1; dx++ )
               {
                  const vec2 dpos( x * 2.f + dx, y * 2.f + dy );
                  const vec4 dH = cv_utils::imsample32F<vec4>( dPyr.back(), dpos );
                  if ( dH.y > 0.0 )
                  {
                     const float dist = distance( pos, dpos );
                     const float z = exp( -dist * dist / sp_z );
                     w += z;
                     val += z * dH;
                  }
               }
            }

            depthPtr[x] = val / ( w > 0.0 ? w : 1.0f );
            assert( !isnan(depthPtr[x].y) );
            assert( depthPtr[x].x <= 1.0 );
         }
      }

      dPyr.push_back( dCurr.clone() );
      iPyr.push_back( iCurr.clone() );
      currPyrSz = std::min( iCurr.rows, iCurr.cols );
   }

   // upscale 
   for ( size_t i = 2; i <= dPyr.size(); ++i )
   {
      const size_t cl = dPyr.size() - i;
      const size_t pl = cl + 1;

      const Mat& iCurr = iPyr[cl];
      const Mat& iPrev = iPyr[pl];

      Mat& dCurr = dPyr[cl];
      const Mat& dPrev = dPyr[pl];

#pragma omp parallel for
      for ( unsigned y = 0; y < dCurr.rows; y++ )
      {
         vec4* depthPtr = dCurr.ptr<vec4>( y );
         const vec3* imgPtr = iCurr.ptr<vec3>( y );

         for ( unsigned x = 0; x < dCurr.cols; x++ )
         {
            vec4& dH = depthPtr[x];

            if ( ( dH.y == 0.0 ) || ( filter > 0.0 ) )
            {
               vec4 val( 0.0 );
               float w = 0.0;

               const vec3& cH = imgPtr[x];
               const vec2 pos( x, y );

               for ( int dy = -1; dy <= 1; dy++ )
               {
                  for ( int dx = -1; dx <= 1; dx++ )
                  {
                     const vec2 dpos( (float)x + (float)dx, (float)y + (float)dy );
                     const vec4 dL = cv_utils::imsample32F<vec4>( dPrev, 0.5f * dpos );
                     if ( isnan(dL.y) || (dL.y == 0.0) )
                     {
                        vec4 dL2 = cv_utils::imsample32F<vec4>( dPrev, 0.5f * dpos );
                     }
                     assert( !isnan(dL.y) );
                     assert( dL.y > 0.0 );
                     const vec3 cL = cv_utils::imsample32F<vec3>( iPrev, 0.5f * dpos );

                     const float sp_dist = distance( pos, dpos );
                     const float col_dist = distance( cL, cH );

                     const float z = exp( -sp_dist * sp_dist / sp_z ) * exp( -col_dist * col_dist / col_z );
                     w += z;
                     assert( !isnan(w) );
                     val += z * dL;
                  }
               }

               dH = (dH.y == 0.0) ? val / w : mix( dH, val / w, filter);
               assert( !isnan(dH.y) );
               assert( dH.y > 0.0 );
               assert( dH.x <= 1.0 );
            }
         }
      }
   }

#pragma omp parallel for
   for ( unsigned y = 0; y < depth.rows; y++ )
   {
      float* depthPtr = depth.ptr<float>( y );
      vec4* dPtr = dPyr.front().ptr<vec4>( y );
      for ( unsigned x = 0; x < depth.cols; x++ )
      {
         depthPtr[x] = dPtr[x].x;
      }
   }
}

//------------------------------------------------------------------------------
//
void processDepth(
    Mat& depth,
    const Mat& img,
    const float undef = 0.0,
    const int rad = 5,
    const float sigCol = 0.1 )
{
   depth =
       depth( Rect( 150, depth.rows / 3 + 20, depth.cols - 300, 2 * depth.rows / 3 - 20 ) ).clone();

   Mat depthOut = depth.clone();

   const uvec2 maxTileSz( 64, 64 );
   const uvec2 nTiles( depth.cols / maxTileSz.x + 1, depth.rows / maxTileSz.y + 1 );
   const vec2 tileSz( (float)depth.cols / nTiles.x, (float)depth.rows / nTiles.y );

   for ( unsigned tileY = 0; tileY < nTiles.y; ++tileY )
   {
      for ( unsigned tileX = 0; tileX < nTiles.x; ++tileX )
      {
         const vec2 tileId( tileX, tileY );
         const vec2 tileStart = max( tileId * tileSz - (float)rad, vec2( 0.0f ) );
         const vec2 tileEnd =
             min( ( tileId + 1.0f ) * tileSz + (float)rad, vec2( depth.cols - 1, depth.rows - 1 ) );

         const Rect tile(
             tileStart.x, tileStart.y, tileEnd.x - tileStart.x + 1, tileEnd.y - tileStart.y + 1 );

         Mat tileDepth = depth( tile );
         Mat tileDepthOut = depthOut( tile );
         Mat tileImg = img( tile );

         /*imshow("depth", depth);
         imshow("tile", tileDepth);
         waitKey(0);*/

         // ------------
         // solve the system || Wdx - Wdb ||^2
         // W is the affinitiy matrix between pixels :
         // w(x,y) = exp( -|I(x)-I(y)|^2 / sig^2 ) if ||x-y||^2 < th
         //        = 0 otherwise
         // dx are the unknown depth values
         // db are the known depth values

         // set the undefined depth position
         std::vector<uvec2> undefIdx;
         {
            vector<vector<uvec2> > yxUndefIdx( tileDepth.rows );
            vector<unsigned> cumUndef( tileDepth.rows );
            //#pragma omp parallel for
            for ( unsigned y = 0; y < tileDepth.rows; y++ )
            {
               const float* depthPtr = tileDepth.ptr<float>( y );
               vector<uvec2>& xUndefIdx = yxUndefIdx[y];
               unsigned& nUndef = cumUndef[y];
               nUndef = 0;
               xUndefIdx.reserve( tileDepth.cols );
               for ( size_t x = 0; x < tileDepth.cols; x++ )
               {
                  if ( depthPtr[x] != undef ) continue;
                  xUndefIdx.emplace_back( x, y );
                  nUndef++;
               }
            }
            for ( unsigned y = 1; y < tileDepth.rows; y++ ) cumUndef[y] += cumUndef[y - 1];
            undefIdx.resize( cumUndef.back() );
            //#pragma omp parallel for
            for ( unsigned y = 0; y < tileDepth.rows - 1; y++ )
            {
               std::cout << undefIdx.size() << "/" << cumUndef[y] << "/" << yxUndefIdx[y].size()
                         << endl;
               memcpy(
                   &undefIdx[cumUndef[y]],
                   &yxUndefIdx[y][0],
                   sizeof( uvec2 ) * yxUndefIdx[y].size() );
            }
         }

         // create the system
         const unsigned nUndef = undefIdx.size();
         MatrixXf W = MatrixXf::Zero( nUndef, nUndef );
         MatrixXf B = MatrixXf::Zero( nUndef, 1 );
         //#pragma omp parallel for
         for ( unsigned u = 0; u < nUndef; ++u )
         {
            const uvec2& upos = undefIdx[u];
            const vec3 ucol = tileImg.at<vec3>( upos.x, upos.y );

            // compute B from all neigboring defined pixels
            float sb_w = 0.0;
            float wb = 0.0;
            W( u, u ) = -1.0;
            for ( int ny = std::max( (int)( upos.y - rad ), 0 );
                  ny <= std::min( (int)( upos.y + rad ), tileDepth.rows - 1 );
                  ++ny )
            {
               for ( int nx = std::max( (int)( upos.x - rad ), 0 );
                     nx <= std::min( (int)( upos.x + rad ), tileDepth.cols - 1 );
                     ++nx )
               {
                  const uvec2 vpos( nx, ny );
                  if ( vpos == upos ) continue;
                  const float vd = tileDepth.at<float>( vpos.x, vpos.y );
                  if ( vd != undef )
                  {
                     const vec3 dcol = tileImg.at<vec3>( vpos.x, vpos.y ) - ucol;
                     const float w = exp( -1.0 * dot( dcol, dcol ) / sigCol );
                     sb_w += w;
                     wb += w * vd;
                  }
               }
            }
            B( u, 0 ) = sb_w > 0.0 ? wb / sb_w : 0.0;

            // compute W from all neigboring undefined pixels
            float sw_w = 0.0;
            for ( unsigned v = 0; v < nUndef; ++v )
            {
               if ( v == u ) continue;
               const uvec2& vpos = undefIdx[v];
               if ( distance( vec2( upos ), vec2( vpos ) ) > rad ) continue;
               const vec3 dcol = tileImg.at<vec3>( vpos.x, vpos.y ) - ucol;
               const float w = exp( -1.0 * dot( dcol, dcol ) / sigCol );
               sw_w += w;
               W( u, v ) = w;
            }
            if ( sw_w > 0.0 ) W.row( u ) = W.row( u ) / sw_w;
         }

         // solve the system
         MatrixXf WtW( nUndef, nUndef );
         WtW.template triangularView<Lower>() = W.transpose() * W;
         MatrixXf WtB = W.transpose() * B;
         WtW.ldlt().solveInPlace( WtB );

         // set the missing value
         //#pragma omp parallel for
         for ( unsigned u = 0; u < nUndef; ++u )
         {
            const uvec2& pos = undefIdx[u];
            tileDepthOut.at<float>( pos.x, pos.y ) = WtB( u, 0 );
         }

         imshow( "tile", tileDepth );
         imshow( "outtile", tileDepthOut );
         waitKey( 0 );
      }
   }

   depth = depthOut;
}
}
//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@imgFileLst   |         | images list   }"
    "{@imgRootDir   |         | images root dir   }"
    "{@imgOutDir    |         | images output dir   }"
    "{@nbSamples      |         |    }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }

   const bool toLinear = false;
   int nbSamples = parser.get<int>( "@nbSamples" );

   const filesystem::path outRootPath( parser.get<string>( "@imgOutDir" ) );

   // Create the list of image triplets

   ImgNFileLst<6> imgLst(
       parser.get<string>( "@imgFileLst" ).c_str(), parser.get<string>( "@imgRootDir" ).c_str() );
   if ( imgLst.size() == 0 )
   {
      cerr << "Invalid dataset : " << parser.get<string>( "@imgFileLst" ) << endl;
      return -1;
   }

   // Loop through the data
   nbSamples = nbSamples <= 0 ? imgLst.size() : nbSamples;
   for ( size_t i = 0; i < nbSamples; ++i )
   {
      const auto& data = imgLst[i];

      const string outPrefix = filesystem::path( data[1] ).parent_path().stem().string() + "_";

      for ( size_t j = 0; j < 3; ++j )
      {
         // load the current image
         Mat img = cv_utils::imread32FC3( data[j * 2] );
         Mat depth = cv_utils::imread32FC1( data[j * 2 + 1] );
         normalize(depth, depth, 1.0, 0.0, NORM_MINMAX);

         processImg( img );

         //imshow( "InDepth", depth( Rect( 150, depth.rows / 3 + 20, depth.cols - 300, 2 * depth.rows / 3 - 20 ) ) );
         
         processDepth( depth, img, 0.0f, 1.0, 0.1, 0.23  );

         // display
         //imshow( "Img", img );
         //imshow( "Depth", depth );

         const string outBasename = outPrefix + filesystem::path( data[j * 2] ).stem().string();
         const filesystem::path fRight( outBasename + string( "_i" ) + ".png" );
         const filesystem::path fDepth( outBasename + string( "_d" ) + ".png" );

         imwrite( filesystem::path( outRootPath / fRight ).string().c_str(), img * 255.0 );
         imwrite( filesystem::path( outRootPath / fDepth ).string().c_str(), depth * 255.0 );

         cout << fRight.string() << " " << fDepth.string();
         if ( j == 2 )
            cout << endl;
         else
            cout << " ";

         //waitKey( 0 );
      }

      // waitKey( 0 );
   }

   return ( 0 );
}
