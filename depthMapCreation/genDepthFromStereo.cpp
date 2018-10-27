/*!
 * *****************************************************************************
 *   \file genDepthFromSBS.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-10-01
 *   *****************************************************************************/

#include "utils/imgFileLst.h"
#include "utils/cv_utils.h"

#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/ximgproc/disparity_filter.hpp"

#include <libopticalFlow/oclVarOpticalFlow.h>

#include <glm/glm.hpp>

#include <boost/filesystem.hpp>

#include <iostream>

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
bool rectifyStereoPair(
    Mat& matRight,
    Mat& matLeft
)
{
  // Points detection
  const int minHessian = 400;
  Ptr<ORB> detector = ORB::create(1500, 1.2f, 12, 31,0,2, ORB::HARRIS_SCORE, 31, 1);
  vector<KeyPoint> arrKPtsRight, arrKPtsLeft;
  Mat matDescRigth, matDescLeft;
  detector->detect( matRight, arrKPtsRight, matDescRigth );
  detector->detect( matLeft, arrKPtsLeft, matDescLeft );
  
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  vector< DMatch > matches;
  matcher->match( matDescRigth, matDescLeft, matches );

  matcher->knnMatch(first_desc, desc, matches, 2);
    for(unsigned i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
            matched1.push_back(first_kp[matches[i][0].queryIdx]);
            matched2.push_back(      kp[matches[i][0].trainIdx]);
        }
    }
  
  //cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3, 0.99);
  Mat F = findFundamentalMat(arrPtsRight, arrPtsLeft, CV_FM_8POINT);
  
  Mat HRight(4,4, CV_32F);
  Mat HLeft(4,4, CV_32F);
  stereoRectifyUncalibrated(arrPtsRight, arrPtsLeft, F, matRight.size(), HRight, HLeft);
  
  warpPerspective(matRight, matRight, HRight, matRight.size());
  warpPerspective(matLeft, matLeft, HLeft, matLeft.size());

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

   return masked < 0.65;
}

//------------------------------------------------------------------------------
//
/*bool flowToDispOCV(
    const Mat&,
    const Mat&,
    const Mat& imgFrom,
    const Mat& imgTo,
    Mat& disp,
    Mat& mask,
    const bool isInverted )
{
   auto left_matcher = StereoBM::create( 16, 13 );
   auto right_matcher = ximgproc::createRightMatcher( left_matcher );
   
   auto wls_filter = createDisparityWLSFilter( left_matcher );

   Mat matRight = 255.0 * imgTo;
   matRight = matRight.convertTo(CV_8UC3);
   Mat matRightGray;
   cvtColor( matRight, matRightGray, COLOR_BGR2GRAY );
   Mat matLeft = 255.0 * imgFrom;
   matLeft = matLeft.convertTo(CV_8UC3);
   Mat matLeftGray;
   cvtColor( imgFrom, matLeft, COLOR_BGR2GRAY );

   Mat matDispLeft(matLeftGray.rows, matLeftGray.cols, CV_16S);
   left_matcher->compute( matLeftGray, matRightGray, matDispLeft );
   Mat matDispRight(matRightGray.rows, matRightGray.cols, CV_16S)
   right_matcher->compute( matRightGray, matLeftGray, matDispRight );

   wls_filter->setLambda( lambda );
   wls_filter->setSigmaColor( sigma );
   wls_filter->filter( matDispLeft, matLeft, disp, matDispRight );

   disp = disp.convertTo(CV_32FC1);
   mask = Mat( flowR.rows, flowR.cols, CV_8UC1, 1 );
}*/
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
         imshow( "Mask", matMask );
         waitKey( 0 );
      }
   }

   return ( 0 );
}
