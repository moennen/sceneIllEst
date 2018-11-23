/*!
 * *****************************************************************************
 *   \file genDepthFromSBS.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-10-01
 *   *****************************************************************************/

#include "utils/imgFileLst.h"
#include "utils/cv_utils.h"

#include <opencv2/features2d/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include <libopticalFlow/oclVarOpticalFlow.h>

#include <glm/glm.hpp>

#include <boost/filesystem.hpp>

#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace glm;
using namespace boost;

namespace
{
//------------------------------------------------------------------------------
//
void resizeToMax( Mat& img, const uvec2 sampleSz )
{
   uvec2 imgSz( img.cols, img.rows );
   // random rescale
   const float ds = std::max( (float)sampleSz.y / imgSz.y, (float)sampleSz.x / imgSz.x );
   if ( ds < 1.0 ) resize( img, img, Size(), ds, ds, CV_INTER_AREA );
}

//------------------------------------------------------------------------------
//
Mat depthToImg( const Mat& depth )
{
   Mat depth0 = depth.clone() * 255.0;
   Mat depth1;
   depth0.convertTo( depth1, CV_8UC1 );
   applyColorMap( depth1, depth0, COLORMAP_JET );
   return depth0;
}

//------------------------------------------------------------------------------
//
Mat flowToWrite( const Mat& flow )
{
   vector<Mat> splitFlow( 3 );
   split( flow, &splitFlow[0] );
   splitFlow[2] = Mat( flow.rows, flow.cols, CV_32FC1, 0.0f );
   Mat out( flow.rows, flow.cols, CV_32FC3 );
   merge( &splitFlow[0], 3, out );
   return out;
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

      normalize( bgr, bgr, 0.0, 1.0, NORM_MINMAX, -1 );
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
Mat flowToErr( const Mat& flow, const Mat& imgFrom, const Mat& imgTo )
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
         lk_data[x] = distance( from, to );
      }
   }

   return lk;
}

//------------------------------------------------------------------------------
//
bool rectifyStereoPair( Mat& matRight, Mat& matLeft )
{
   cvtColor( matRight, matRight, COLOR_RGB2BGR );
   cvtColor( matLeft, matLeft, COLOR_RGB2BGR );

   // Points detection
   const int minHessian = 400;
   vector<KeyPoint> arrKPtsRight, arrKPtsLeft;
   Mat matDescRight, matDescLeft;
   Ptr<ORB> detector = ORB::create( 1750, 1.2f, 4, 31, 0, 2, ORB::HARRIS_SCORE, 31, 10 );
   detector->detectAndCompute( matRight, noArray(), arrKPtsRight, matDescRight );
   detector->detectAndCompute( matLeft, noArray(), arrKPtsLeft, matDescLeft );

   Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce-Hamming" );
   vector<vector<DMatch>> matchess;
   matcher->radiusMatch( matDescRight, matDescLeft, matchess, 50.0f );

   std::cout << arrKPtsRight.size() << " / " << arrKPtsLeft.size() << " --> " << matchess.size()
             << " " << matchess[0].size() << endl;

   vector<DMatch> matches;
   matches.reserve( matchess.size() );
   for ( const auto m : matchess )
   {
      if ( !m.empty() ) matches.push_back( m.front() );
   }

   const size_t best_k = std::min( (size_t)75u, matches.size() );
   if ( matches.size() > best_k )
   {
      sort( matches.begin(), matches.end(), []( const DMatch& a, const DMatch& b ) {
         return a.distance < b.distance;
      } );
   }

   vector<Point2f> arrPtsRight( best_k );
   vector<Point2f> arrPtsLeft( best_k );
   for ( size_t k = 0; k < best_k; ++k )
   {
      arrPtsRight[k] = arrKPtsRight[matches[k].queryIdx].pt;
      arrPtsLeft[k] = arrKPtsLeft[matches[k].trainIdx].pt;
   }

   //-- Draw matches
   matches.resize( best_k );
   Mat img_matches;
   drawMatches( matRight, arrKPtsRight, matLeft, arrKPtsLeft, matches, img_matches );
   imshow( "Matches", img_matches );

   waitKey( 0 );

   // Mat F = findFundamentalMat(arrPtsRight, arrPtsLeft, FM_RANSAC, 2.5, 0.99);
   Mat F = findFundamentalMat( arrPtsRight, arrPtsLeft, CV_FM_8POINT );

   cout << "Fundamental : " << endl;
   cout << F << endl;

   Mat HRight( 4, 4, CV_64F );
   Mat HLeft( 4, 4, CV_64F );
   stereoRectifyUncalibrated( arrPtsRight, arrPtsLeft, F, matRight.size(), HRight, HLeft );

   HLeft = HRight.inv() * HLeft;
   HRight = Mat::eye( HRight.rows, HRight.cols, CV_64F );

   cout << "Homo : " << endl;
   cout << HLeft << endl;

   warpPerspective( matRight, matRight, HRight, matRight.size() );
   warpPerspective( matLeft, matLeft, HLeft, matLeft.size() );

   imshow( "MatRight", matRight );
   imshow( "MatLeft", matLeft );

   cvtColor( matRight, matRight, COLOR_BGR2RGB );
   cvtColor( matLeft, matLeft, COLOR_BGR2RGB );

   return true;
}

//------------------------------------------------------------------------------
//
bool rectifyStereoPairOf( Mat& matRight, Mat& matLeft, OclVarOpticalFlow& ofEst )
{
   // compute the left to right optical flow
   Mat matOfL2R( matLeft.rows, matLeft.cols, CV_32FC2 );
   ofEst.compute(
       reinterpret_cast<const float*>( matRight.ptr() ),
       reinterpret_cast<const float*>( matLeft.ptr() ),
       matLeft.cols,
       matLeft.rows,
       reinterpret_cast<float*>( matOfL2R.ptr() ) );

   // create matches

   const ivec2 grid_res = {matRight.cols / 33, matRight.rows / 33};
   const vec2 delta = {matRight.cols / ( grid_res.x + 1 ), matRight.rows / ( grid_res.y + 1 )};

   vector<DMatch> matches;
   matches.reserve( grid_res.x * grid_res.y );
   vector<KeyPoint> arrKPtsRight;
   arrKPtsRight.reserve( grid_res.x * grid_res.y );
   vector<KeyPoint> arrKPtsLeft;
   arrKPtsLeft.reserve( grid_res.x * grid_res.y );

   for ( int j = 0; j < grid_res.y; ++j )
   {
      const float y = (j+1) * delta.y;
      for ( int i = 0; i < grid_res.x; ++i )
      {
         const size_t idx = j * grid_res.x + i;
         const float x = (i+1) * delta.x;
         const vec2 pos = {x, y};
         const vec2 uv = cv_utils::imsample32F<vec2>( matLeft, pos );

         const vec3 rgb_l = cv_utils::imsample32F<vec3>( matLeft, pos );
         const vec3 rgb_r = cv_utils::imsample32F<vec3>( matRight, pos + uv );

         matches.emplace_back( arrKPtsRight.size(), arrKPtsLeft.size(), distance( rgb_l, rgb_r ) );
         arrKPtsRight.emplace_back( Point2f( pos.x + uv.x, pos.y + uv.y ), 3.0 );
         arrKPtsLeft.emplace_back( Point2f( pos.x, pos.y ), 3.0f );
      }
   }


   cvtColor( matRight, matRight, COLOR_BGR2RGB );
   cvtColor( matLeft, matLeft, COLOR_BGR2RGB );

   //-- Draw matches
   Mat img_matches;
   drawMatches( matRight, arrKPtsRight, matLeft, arrKPtsLeft, matches, img_matches );
   imshow( "Matches", img_matches );

   vector<Point2f> arrPtsRight( arrKPtsRight.size() );
   for ( size_t k = 0; k < arrKPtsRight.size(); ++k )  arrPtsRight[k] = arrKPtsRight[k].pt;
   vector<Point2f> arrPtsLeft( arrKPtsLeft.size() );
   for ( size_t k = 0; k < arrKPtsLeft.size(); ++k )  arrPtsLeft[k] = arrKPtsLeft[k].pt;

   // Mat F = findFundamentalMat(arrPtsRight, arrPtsLeft, FM_RANSAC, 2.5, 0.99);
   Mat F = findFundamentalMat( arrPtsRight, arrPtsLeft, CV_FM_8POINT );

   cout << "Fundamental : " << endl;
   cout << F << endl;

   Mat HRight( 4, 4, CV_64F );
   Mat HLeft( 4, 4, CV_64F );
   stereoRectifyUncalibrated( arrPtsRight, arrPtsLeft, F, matRight.size(), HRight, HLeft );

   HLeft = HRight.inv() * HLeft;
   HRight = Mat::eye( HRight.rows, HRight.cols, CV_64F );

   cout << "Homo : " << endl;
   cout << HLeft << endl;

   //warpPerspective( matRight, matRight, HRight, matRight.size() );
   warpPerspective( matLeft, matLeft, HLeft, matLeft.size() );

   imshow( "MatRight", matRight );
   imshow( "MatLeft", matLeft );

   waitKey( 0 );

   cvtColor( matRight, matRight, COLOR_BGR2RGB );
   cvtColor( matLeft, matLeft, COLOR_BGR2RGB );

   return true;
}

//------------------------------------------------------------------------------
//
bool flowToDisp(
    const Mat& flowR,
    const Mat& flowL,
    const Mat& imgFrom,
    const Mat& imgTo,
    Mat& disp,
    Mat& mask,
    const bool isInverted )
{
   disp = Mat( flowR.rows, flowR.cols, CV_32FC1 );
   mask = Mat( flowR.rows, flowR.cols, CV_8UC1 );

   Mat errR = flowToErr( flowR, imgFrom, imgTo );

#pragma omp parallel for
   for ( size_t y = 0; y < flowR.rows; y++ )
   {
      const vec2* f_row_r_data = flowR.ptr<vec2>( y );
      const float* err_data = errR.ptr<float>( y );
      const vec2* f_row_l_data = flowL.ptr<vec2>( y );

      float* d_row_data = disp.ptr<float>( y );
      unsigned char* u_row_data = mask.ptr<unsigned char>( y );

      for ( size_t x = 0; x < flowR.cols; x++ )
      {
         const vec2 mtR = f_row_r_data[x];
         const float dispR = sign( mtR.x ) * length( mtR );
         const vec2 mtRL = vec2( -1.0 ) * cv_utils::imsample32F<vec2>( flowL, vec2( x, y ) + mtR );
         const float dispRL = sign( mtRL.x ) * length( mtRL );

         const float dispReg = mix( dispR, dispRL, 0.5 );
         const bool undef = ( abs( mtR.y ) > 1.0 ) || ( abs( mtRL.y ) > 1.0 ) ||
                            ( err_data[x] > 0.5 ) || ( distance( mtR, mtRL ) > 1.0 );

         u_row_data[x] = undef ? 0 : 255;
         d_row_data[x] = ( isInverted ? 1.0 : -1.0 ) * dispReg;
      }
   }

   normalize( disp, disp, 0.0, 1.0, NORM_MINMAX, -1, mask );

   const float masked = ( 1.0f - ( cv::mean( mask )[0] / 255.0f ) );

   return true;  // masked < 0.65;
}
}

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@imgFileLst    |         | images list   }"
    "{@imgRootDir    |         | images root dir   }"
    "{@imgOutDir     |         | images output dir   }"
    "{outWidth       |640      | output width }"
    "{outHeight      |380      | output height }"
    "{show           |         |    }"
    "{nowrite        |         |    }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }

   const uvec2 minSz =
       uvec2( parser.get<unsigned>( "outWidth" ), parser.get<unsigned>( "outHeight" ) );

   const bool toLinear = false;
   const bool doShow = parser.get<bool>( "show" );
   const bool doWrite = !parser.get<bool>( "nowrite" );

   const filesystem::path outRootPath( parser.get<string>( "@imgOutDir" ) );

   // Create the list of image triplets
   ImgNFileLst imgLst(
       2,
       parser.get<string>( "@imgFileLst" ).c_str(),
       parser.get<string>( "@imgRootDir" ).c_str() );
   if ( imgLst.size() == 0 )
   {
      cerr << "Invalid dataset : " << parser.get<string>( "@imgFileLst" ) << endl;
      return -1;
   }

   // Create the optical flow estimator
   OclVarOpticalFlow::params_t ofParams = OclVarOpticalFlow::getDefaultParams();
   ofParams.nonLinearIter = 9;
   ofParams.robustIter = 5;
   ofParams.solverIter = 5;
   ofParams.lambda = 0.1;
   ofParams.gamma = 75;

   // NB : need to disable OpenCV/OpenCL by setting the envvar :
   // setenv OPENCV_OPENCL_RUNTIME 0
   OclVarOpticalFlow ofEstimator( minSz.x, minSz.y, false, ofParams );

   for ( size_t i = 0; i < imgLst.size(); ++i )
   {
      const string outBasename = filesystem::path( imgLst( i, 0 ) ).stem().string();
      const string videoIdname = outBasename.substr( 0, outBasename.find_first_of( "_" ) );

      Mat matRight = cv_utils::imread32FC3( imgLst.filePath( i, 0 ), toLinear, true );
      Mat matLeft = cv_utils::imread32FC3( imgLst.filePath( i, 1 ), toLinear, true );
      if ( matRight.empty() || matLeft.empty() ) continue;

      resizeToMax( matRight, minSz );
      resizeToMax( matLeft, minSz );

      // compute the right to left optical flow
      ofEstimator.setImgSize( matRight.cols, matRight.rows );

      rectifyStereoPairOf( matRight, matLeft, ofEstimator );

      Mat matOfR2L( matRight.rows, matRight.cols, CV_32FC2 );
      ofEstimator.compute(
          reinterpret_cast<const float*>( matLeft.ptr() ),
          reinterpret_cast<const float*>( matRight.ptr() ),
          matRight.cols,
          matRight.rows,
          reinterpret_cast<float*>( matOfR2L.ptr() ) );

      // compute the left to right optical flow
      Mat matOfL2R( matLeft.rows, matLeft.cols, CV_32FC2 );
      ofEstimator.compute(
          reinterpret_cast<const float*>( matRight.ptr() ),
          reinterpret_cast<const float*>( matLeft.ptr() ),
          matLeft.cols,
          matLeft.rows,
          reinterpret_cast<float*>( matOfL2R.ptr() ) );

      // compute the disparity and disparity mask
      Mat matDepth, matMask;
      if ( !flowToDisp( matOfR2L, matOfL2R, matRight, matLeft, matDepth, matMask, false ) )
         continue;

      cvtColor( matRight, matRight, COLOR_BGR2RGB );
      cvtColor( matLeft, matLeft, COLOR_BGR2RGB );

      const filesystem::path fRight(
          videoIdname + "/" + outBasename + string( "_right_rgb" ) + ".png" );
      const filesystem::path fLeft(
          videoIdname + "/" + outBasename + string( "_left_rgb" ) + ".png" );
      const filesystem::path fFlowR2L(
          videoIdname + "/" + outBasename + string( "_r2l_mflow" ) + ".exr" );
      const filesystem::path fFlowL2R(
          videoIdname + "/" + outBasename + string( "_l2r_mflow" ) + ".exr" );
      const filesystem::path fDepth( videoIdname + "/" + outBasename + string( "_d" ) + ".exr" );
      const filesystem::path fMask( videoIdname + "/" + outBasename + string( "_a" ) + ".png" );

      if ( doWrite )
      {
         imwrite( filesystem::path( outRootPath / fRight ).string().c_str(), matRight * 255.0 );
         imwrite( filesystem::path( outRootPath / fLeft ).string().c_str(), matLeft * 255.0 );
         imwrite(
             filesystem::path( outRootPath / fFlowR2L ).string().c_str(), flowToWrite( matOfR2L ) );
         imwrite(
             filesystem::path( outRootPath / fFlowL2R ).string().c_str(), flowToWrite( matOfL2R ) );
         imwrite( filesystem::path( outRootPath / fDepth ).string().c_str(), matDepth );
         imwrite( filesystem::path( outRootPath / fMask ).string().c_str(), matMask );
      }

      cout << fRight.string() << " " << fLeft.string() << " " << fDepth.string() << " "
           << fMask.string() << " " << fFlowR2L.string() << " " << fFlowL2R.string() << endl;

      // display
      if ( doShow )
      {
         imshow( "Right", matRight );
         imshow( "Left", matLeft );
         imshow( "FlowR2L", flowToImg( matOfR2L, true ) );
         imshow( "FlowL2R", flowToImg( matOfL2R, true ) );
         imshow( "Disp", depthToImg( matDepth ) );
         imshow( "Depth", matDepth );
         imshow( "Mask", matMask );
         waitKey( 0 );
      }
   }

   return ( 0 );
}
