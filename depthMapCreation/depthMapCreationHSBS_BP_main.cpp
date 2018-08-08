/*!
 * *****************************************************************************
 *   \file depthMapCreationHSBS_main.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-02-19
 *   *****************************************************************************/

#include "utils/imgFileLst.h"
#include "utils/cv_utils.h"

#include <libopticalFlow/oclVarOpticalFlow.h>

#include "opencv2/cudastereo.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <boost/filesystem.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;
using namespace boost;

namespace
{
const std::array<std::string, 7> stripedVideoIdName = {
    {"0001", "0005", "0006", "0007", "0009", "0015", "0018"}};
const std::array<std::string, 1> invertVideoIdName = {{"0010"}};

//------------------------------------------------------------------------------
//
void resizeToMin( Mat& img, const uvec2 sampleSz )
{
   uvec2 imgSz( img.cols, img.rows );
   // random rescale
   const float ds = std::min( (float)sampleSz.y / imgSz.y, (float)sampleSz.x / imgSz.x );
   resize( img, img, Size(), ds, ds, CV_INTER_AREA );
}

//------------------------------------------------------------------------------
//
void split_hsbs( const Mat& img, Mat& right, Mat& left, const unsigned nHStripes = 0 )
{
   const uvec2 off = {0, 0};
   const uvec2 hSz = {img.cols / 2, img.rows - 2 * nHStripes};
   right = img( Rect( off.x, nHStripes + off.y, hSz.x - 2 * off.x, hSz.y - 2 * off.y ) ).clone();
   left = img( Rect( hSz.x + off.x, nHStripes + off.y, hSz.x - 2 * off.x, hSz.y - 2 * off.y ) )
              .clone();
}

//------------------------------------------------------------------------------
//
void compute_disparity( const Mat& right, const Mat& left, Mat& disp )
{
   cv::cuda::printShortCudaDeviceInfo( cv::cuda::getDevice() );

   Ptr<cuda::StereoBeliefPropagation> bp = cuda::createStereoBeliefPropagation( 512 );

   Mat gleft, gright;
   cvtColor( left * 255.0f, gleft, COLOR_BGR2GRAY );
   gleft.convertTo( gleft, CV_8U );
   cvtColor( right * 255.0f, gright, COLOR_BGR2GRAY );
   gright.convertTo( gright, CV_8U );

   cuda::GpuMat d_left, d_right;
   d_left.upload( gleft );
   d_right.upload( gright );

   disp = Mat( left.size(), CV_8U );
   cuda::GpuMat d_disp( left.size(), CV_8U );

   bp->compute( d_left, d_right, d_disp );

   d_disp.download( disp );
}

//------------------------------------------------------------------------------
//
Mat stereoBlendView( const Mat& R, const Mat& L )
{
   Mat gR, gL;
   cvtColor( R, gR, COLOR_RGB2GRAY );
   cvtColor( L, gL, COLOR_RGB2GRAY );

   Mat gRSplit[3], gLSplit[3], resSplit[3], res;
   split( gR, gRSplit );
   resSplit[1] = gRSplit[0];
   split( gL, gLSplit );
   resSplit[2] = gLSplit[0];
   resSplit[0] = Mat::zeros( resSplit[1].size(), resSplit[1].type() );
   merge( resSplit, 3, res );

   return res;
}

//------------------------------------------------------------------------------
//
Mat flowToImg( const Mat& flow, const bool leg = false )
{
   Mat flow_split[2];
   Mat t_split[3];
   Mat bgr;
   split( flow, flow_split );
   if ( leg )
   {
      t_split[0] = Mat::zeros( flow_split[0].size(), flow_split[0].type() );
      t_split[1] = flow_split[1];
      t_split[2] = flow_split[0];
      merge( t_split, 3, bgr );
   }
   else
   {
      Mat hsv;
      Mat magnitude, angle;
      cartToPolar( flow_split[0], flow_split[1], magnitude, angle, true );
      normalize( magnitude, magnitude, 0, 1, NORM_MINMAX );
      t_split[0] = angle;  // already in degrees - no normalization needed
      t_split[1] = Mat::ones( angle.size(), angle.type() );
      t_split[2] = magnitude;
      merge( t_split, 3, hsv );
      cvtColor( hsv, bgr, COLOR_HSV2BGR );
   }
   return bgr;
}

//------------------------------------------------------------------------------
//
Mat flowToLk( const Mat& flow, const Mat& imgFrom, const Mat& imgTo )
{
   Mat lk( flow.rows, flow.cols, CV_32FC1 );

#pragma omp parallel for
   for ( size_t y = 0; y < flow.rows; y++ )
   {
      float* lk_data = lk.ptr<float>( y );
      const vec2* flow_data = flow.ptr<vec2>( y );
      const vec3* from_data = imgFrom.ptr<vec3>( y );

      for ( size_t x = 0; x < flow.cols; x++ )
      {
         const vec2 f = flow_data[x];
         const vec3 from = from_data[x];
         const vec3 to = cv_utils::imsample32FC3<vec3>( imgTo, vec2( x, y ) + f );

         vec3 dist = from - to;

         lk_data[x] = exp( -dot( dist, dist ) / 0.33 );
      }
   }

   return lk;
}

//------------------------------------------------------------------------------
//
void processDepth(
    Mat& depth,
    const Mat& img,
    const Mat& undef,
    const float sp_z,
    const float col_z,
    const float filter )
{
   vector<Mat> dPyr;
   dPyr.emplace_back( depth.rows, depth.cols, CV_32FC4 );

   // fill the first level
#pragma omp parallel for
   for ( unsigned y = 0; y < depth.rows; y++ )
   {
      const float* depthPtr = depth.ptr<float>( y );
      const float* undefPtr = undef.ptr<float>( y );
      vec4* dPtr = dPyr.back().ptr<vec4>( y );
      for ( unsigned x = 0; x < depth.cols; x++ )
      {
         const float d = depthPtr[x];
         const bool isUndef = undefPtr[x] < 0.5;
         dPtr[x] = isUndef ? vec4( 0.0 )
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
            assert( !isnan( depthPtr[x].y ) );
            // assert( depthPtr[x].x <= 1.0 );
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
                     if ( isnan( dL.y ) || ( dL.y == 0.0 ) )
                     {
                        vec4 dL2 = cv_utils::imsample32F<vec4>( dPrev, 0.5f * dpos );
                     }
                     assert( !isnan( dL.y ) );
                     assert( dL.y > 0.0 );
                     const vec3 cL = cv_utils::imsample32F<vec3>( iPrev, 0.5f * dpos );

                     const float sp_dist = distance( pos, dpos );
                     const float col_dist = distance( cL, cH );

                     const float z =
                         exp( -sp_dist * sp_dist / sp_z ) * exp( -col_dist * col_dist / col_z );
                     w += z;
                     assert( !isnan( w ) );
                     val += z * dL;
                  }
               }

               dH = ( dH.y == 0.0 ) ? val / w : mix( dH, val / w, filter );
               assert( !isnan( dH.y ) );
               assert( dH.y > 0.0 );
               // assert( dH.x <= 1.0 );
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
Mat flowToDisp(
    const Mat& flowR,
    const Mat& flowL,
    const Mat& imgFrom,
    const Mat& imgTo,
    const bool isInverted )
{
   Mat disp( flowR.rows, flowR.cols, CV_32FC1 );
   Mat undef( flowR.rows, flowR.cols, CV_32FC1 );

   Mat lkR = flowToLk( flowR, imgFrom, imgTo );
   Mat lkL = flowToLk( flowL, imgTo, imgFrom );

   // Mat flowLi( flowR.rows, flowR.cols, CV_32FC2 );

#pragma omp parallel for
   for ( size_t y = 0; y < flowR.rows; y++ )
   {
      const float* f_row_r_data = flowR.ptr<float>( y );
      const float* lk_r_data = lkR.ptr<float>( y );
      const float* f_row_l_data = flowL.ptr<float>( y );
      const float* lk_l_data = lkL.ptr<float>( y );

      // vec2* f_row_li_data = flowLi.ptr<vec2>(y);

      float* d_row_data = disp.ptr<float>( y );
      float* u_row_data = undef.ptr<float>( y );

      for ( size_t x = 0; x < flowR.cols; x++ )
      {
         const float dispR = f_row_r_data[x * 2];
         const float dispRL = -1.0 * mix( f_row_l_data[( x + (int)ceil( dispR ) ) * 2],
                                          f_row_l_data[( x + (int)floor( dispR ) ) * 2],
                                          ceil( dispR ) - dispR );
         // f_row_li_data[x] = vec2(dispRL, 0.0);
         const float dispReg = mix( dispR, dispRL, 0.5 );

         u_row_data[x] =
             ( lk_r_data[x] > 0.5 ? 1.0 : 0.0 ) *
             ( mix( lk_l_data[x + (int)ceil( dispR )],
                    lk_l_data[x + (int)floor( dispR )],
                    ceil( dispR ) - dispR ) > 0.5
                   ? 1.0
                   : 0.0 ) *
             ( exp( -( dispR - dispRL ) * ( dispR - dispRL ) / 3.0 ) > 0.5 ? 1.0 : 0.0 );

         d_row_data[x] = ( isInverted ? 1.0 : -1.0 ) * dispReg;
      }
   }

   /*imshow("flowR", flowToImg(flowR));
   imshow("flowL", flowToImg(flowL));
   imshow("flowLi", flowToImg(flowLi));
   imshow("lkR", lkR);
   imshow("lkL", lkL);
   imshow("undef", undef);*/

   processDepth( disp, imgFrom, undef, 1.0, 0.03, 0.21 );

   normalize( disp, disp, 0.0, 1.0, NORM_MINMAX );

   return disp;
}
}

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@imgFileLst      |         | images list   }"
    "{@imgRootDir    |         | images root dir   }"
    "{@imgOutDir    |         | images output dir   }"
    "{@startIdx    |         |    }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }

   const uvec2 maxSz = uvec2( 640, 480 );
   const bool toLinear = false;
   const float delayOnError = 100;
   const int startIdx = parser.get<int>( "@startIdx" );

   const filesystem::path outRootPath( parser.get<string>( "@imgOutDir" ) );

   // Create the list of image triplets

   ImgNFileLst<1> imgLst(
       parser.get<string>( "@imgFileLst" ).c_str(), parser.get<string>( "@imgRootDir" ).c_str() );
   if ( imgLst.size() == 0 )
   {
      cerr << "Invalid dataset : " << parser.get<string>( "@imgFileLst" ) << endl;
      return -1;
   }

   // Create the optical flow estimator
   OclVarOpticalFlow::params_t ofParams = OclVarOpticalFlow::getDefaultParams();
   ofParams.lambda = 0.19;

   OclVarOpticalFlow ofEstimator( 512, 512, false, ofParams );

   // Loop through the data
   for ( size_t i = startIdx; i < imgLst.size(); ++i )
   {
      const auto& data = imgLst[i];

      // load the current image
      // for ( size_t j = 0; j < 3; ++j )
      const size_t j = 0;
      {
         const string outBasename = filesystem::path( data[j] ).stem().string();
         const string videoIdname = outBasename.substr( 0, outBasename.find_first_of( "_" ) );

         Mat img = cv_utils::imread32FC3( data[j] );
         GaussianBlur( img, img, Size( 3, 3 ), 0.75 );

         const bool isStriped =
             find( stripedVideoIdName.begin(), stripedVideoIdName.end(), videoIdname ) !=
             stripedVideoIdName.end();

         const bool isInverted =
             find( invertVideoIdName.begin(), invertVideoIdName.end(), videoIdname ) !=
             invertVideoIdName.end();

         // split the current image
         Mat right, left;
         split_hsbs( img, right, left, isStriped ? ( img.rows - img.rows / 1.35 ) / 2 : 0 );
         resizeToMin( right, maxSz );
         resizeToMin( left, maxSz );

         ofEstimator.setImgSize( right.cols, right.rows );

         Mat ofRight( right.rows, right.cols, CV_32FC2 );
         /*ofEstimator.setOpt( OclVarOpticalFlow::OptsDoRightDisparity, true );
         ofEstimator.setOpt( OclVarOpticalFlow::OptsDoLeftDisparity, false );*/
         ofEstimator.compute(
             reinterpret_cast<const float*>( left.ptr() ),
             reinterpret_cast<const float*>( right.ptr() ),
             right.cols,
             right.rows,
             reinterpret_cast<float*>( ofRight.ptr() ) );

         // filter on low disparities
         if ( cv::mean( cv::abs( ofRight ) )[0] < 1.0 ) continue;

         Mat ofLeft( right.rows, right.cols, CV_32FC2 );
         /*ofEstimator.setOpt( OclVarOpticalFlow::OptsDoRightDisparity, false );
         ofEstimator.setOpt( OclVarOpticalFlow::OptsDoLeftDisparity, true );*/
         ofEstimator.compute(
             reinterpret_cast<const float*>( right.ptr() ),
             reinterpret_cast<const float*>( left.ptr() ),
             right.cols,
             right.rows,
             reinterpret_cast<float*>( ofLeft.ptr() ) );

         // filter on low disparities
         if ( cv::mean( cv::abs( ofLeft ) )[0] < 1.0 ) continue;

         Mat depth = flowToDisp( ofRight, ofLeft, right, left, isInverted );

         // display
         // imshow( "Full", img );
         /*imshow( "Right", right );
         imshow( "Left", left );
         imshow( "Disp", depth );*/

         const filesystem::path fRight( outBasename + string( "_i" ) + ".png" );
         const filesystem::path fDepth( outBasename + string( "_d" ) + ".exr" );

         imwrite( filesystem::path( outRootPath / fRight ).string().c_str(), right * 255.0 );
         imwrite( filesystem::path( outRootPath / fDepth ).string().c_str(), depth );

         cout << fRight.string() << " " << fDepth.string();

         // if ( j == 2 )
         cout << endl;
         /*else
            cout << " ";*/

         // waitKey( 0 );
      }
   }

   return ( 0 );
}
