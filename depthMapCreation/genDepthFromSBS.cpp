/*!
 * *****************************************************************************
 *   \file genDepthFromSBS.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-10-01
 *   *****************************************************************************/

#include "utils/imgFileLst.h"
#include "utils/cv_utils.h"

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
const double dRatio_16_9 = 16.0 / 9.0;

// SBS Dataset info
// -> video contains horizontal black strips
const std::array<std::string, 10> stripedVideoIdName = {
    {"0000", "0004", "0005", "0006", "0008", "0014", "0017", "0022", "0041", "0046"}};
// -> video contains small horizontal black strips
const std::array<std::string, 4> smallStripedVideoIdName = {{"0003", "0015", "0018"}};
// -> video is not compressed horizontally
const std::array<std::string, 2> uncompressedVideoIdName = {{"0017", "0046"}};
// -> depth is inverted (0 = far)
const std::array<std::string, 2> invertVideoIdName = {{"0009", "0027"}};
// -> video to filter out due to wrong alignement or lack of colour homogeneity between views
const std::array<std::string, 13> ignoreVideoIdName = {{"0022",
                                                        "0023",
                                                        "0024",
                                                        "0026",
                                                        "0027",
                                                        "0028",
                                                        "0034",
                                                        "0035",
                                                        "0036",
                                                        "0038",
                                                        "0041",
                                                        "0046",
                                                        "0048"}};

//------------------------------------------------------------------------------
//
void resizeToMax( Mat& img, const uvec2 sampleSz )
{
   uvec2 imgSz( img.cols, img.rows );
   // random rescale
   const float ds = std::max( (float)sampleSz.y / imgSz.y, (float)sampleSz.x / imgSz.x );
   if (ds < 1.0) resize( img, img, Size(), ds, ds, CV_INTER_AREA );
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
Mat hstrech( const Mat& img, const double dstRatio )
{
   const double fScaleW = dstRatio * img.rows / img.cols;
   const unsigned iScaledW = static_cast<unsigned>(ceil(fScaleW*img.cols));


   Mat simg( img.rows, iScaledW, CV_32FC3 );

   const float fScaleX = static_cast<float>(img.cols) / iScaledW;

#pragma omp parallel for
   for ( size_t y = 0; y < simg.rows; y++ )
   {
      vec3* simg_data = simg.ptr<vec3>( y );

      for ( size_t x = 0; x < simg.cols; x++ )
      {
         simg_data[x] = cv_utils::imsample32FC3<vec3>( img, vec2( fScaleX * x, y ) );
      }
   }

   return simg;
}

//------------------------------------------------------------------------------
//
Mat depthToImg( const Mat& depth )
{
  Mat depth0 = depth.clone() * 255.0;
  Mat depth1; 
  depth0.convertTo(depth1, CV_8UC1);
  applyColorMap(depth1,depth0,COLORMAP_JET);
  return depth0;
}

//------------------------------------------------------------------------------
//
Mat flowToWrite( const Mat& flow )
{
  vector<Mat> splitFlow(3); split(flow, &splitFlow[0]);
  splitFlow[2] = Mat(flow.rows, flow.cols, CV_32FC1, 0.0f);
  Mat out(flow.rows, flow.cols, CV_32FC3);
  merge(&splitFlow[0],3,out);
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
    "{nostretch      |         |    }"
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
   const float delayOnError = 100;
   const bool doShow = parser.get<bool>( "show" );
   const bool doWrite = !parser.get<bool>( "nowrite" );
   const bool nostretch =  parser.get<bool>( "nostretch" );

   const filesystem::path outRootPath( parser.get<string>( "@imgOutDir" ) );

   // Create the list of image triplets
   ImgNFileLst imgLst(
       1,
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

      const bool ignore = find( ignoreVideoIdName.begin(), ignoreVideoIdName.end(), videoIdname ) !=
                          ignoreVideoIdName.end();
      if ( ignore ) continue;

      Mat img = cv_utils::imread32FC3( imgLst.filePath( i, 0 ), toLinear, true );

      const bool uncompressed = nostretch ||
          find( uncompressedVideoIdName.begin(), uncompressedVideoIdName.end(), videoIdname ) !=
          uncompressedVideoIdName.end();

      const bool hasLargeStripes =
          find( stripedVideoIdName.begin(), stripedVideoIdName.end(), videoIdname ) !=
          stripedVideoIdName.end();

      const bool hasSmallStripes =
          !hasLargeStripes &&
          find( smallStripedVideoIdName.begin(), smallStripedVideoIdName.end(), videoIdname ) !=
              smallStripedVideoIdName.end();

      const bool isInverted =
          find( invertVideoIdName.begin(), invertVideoIdName.end(), videoIdname ) !=
          invertVideoIdName.end();

      // split the current image
      Mat right, left;
      split_hsbs(
          img,
          right,
          left,
          hasLargeStripes ? 0.133 * img.rows : ( hasSmallStripes ? 0.0625 * img.rows : 0 ) );

      // if needed we could swap right - left
      if ( isInverted ) swap( right, left );

      if ( !uncompressed ) right = hstrech( right, dRatio_16_9 );
      resizeToMax( right, minSz );
      if ( !uncompressed ) left = hstrech( left, dRatio_16_9 );
      resizeToMax( left, minSz );

      // clone the image for output
      Mat oright = right.clone();
      Mat oleft = left.clone();

      // apply a blur to ease the optical flow estimation
      GaussianBlur( right, right, Size( 3, 3 ), 0.113 );
      GaussianBlur( left, left, Size( 3, 3 ), 0.113 );

      // compute the right to left optical flow
      ofEstimator.setImgSize( right.cols, right.rows );

      Mat ofRight( right.rows, right.cols, CV_32FC2 );
      ofEstimator.compute(
          reinterpret_cast<const float*>( left.ptr() ),
          reinterpret_cast<const float*>( right.ptr() ),
          right.cols,
          right.rows,
          reinterpret_cast<float*>( ofRight.ptr() ) );

      // filter on low disparities
      if ( cv::mean( cv::abs( ofRight ) )[0] < 0.5 ) continue;

      // compute the left to right optical flow
      Mat ofLeft( right.rows, right.cols, CV_32FC2 );
      ofEstimator.compute(
          reinterpret_cast<const float*>( right.ptr() ),
          reinterpret_cast<const float*>( left.ptr() ),
          right.cols,
          right.rows,
          reinterpret_cast<float*>( ofLeft.ptr() ) );

      // filter on low disparities
      if ( cv::mean( cv::abs( ofLeft ) )[0] < 0.5 ) continue;

      // compute the disparity and disparity mask
      Mat depth, mask;
      if ( !flowToDisp(
               ofRight, ofLeft, right, left, depth, mask, false /*rigth-left swapped for
inverted */ ) )
         continue;

      cvtColor(oright, oright, COLOR_BGR2RGB );
      cvtColor(oleft, oleft, COLOR_BGR2RGB );

      const filesystem::path fRight(
          videoIdname + "/" + outBasename + string( "_right_rgb" ) + ".png" );
      const filesystem::path fLeft(
          videoIdname + "/" + outBasename + string( "_left_rgb" ) + ".png" );
      const filesystem::path fFlow( videoIdname + "/" + outBasename + string( "_r2l_mflow" ) + ".exr" );
      const filesystem::path fDepth( videoIdname + "/" + outBasename + string( "_d" ) + ".exr" );
      const filesystem::path fMask( videoIdname + "/" + outBasename + string( "_a" ) + ".png" );

      if ( doWrite )
      {
         imwrite( filesystem::path( outRootPath / fRight ).string().c_str(), oright * 255.0 );
         imwrite( filesystem::path( outRootPath / fLeft ).string().c_str(), oleft * 255.0 );
         imwrite( filesystem::path( outRootPath / fFlow ).string().c_str(), flowToWrite(ofRight) );
         imwrite( filesystem::path( outRootPath / fDepth ).string().c_str(), depth );
         imwrite( filesystem::path( outRootPath / fMask ).string().c_str(), mask );
      }

      cout << fRight.string() << " " << fLeft.string() << " " << fDepth.string() << " "
           << fMask.string() << endl;

      // display
      if ( doShow )
      {
         cvtColor(img, img, COLOR_BGR2RGB );
         imshow( "Full", img );

         imshow( "Right", oright );
         imshow( "Left", oleft );
         imshow( "Flow", flowToImg(ofRight) );
         imshow( "Disp", depthToImg(depth) );
         imshow( "Mask", mask );
         waitKey( 0 );
      }
   }

   return ( 0 );
}
