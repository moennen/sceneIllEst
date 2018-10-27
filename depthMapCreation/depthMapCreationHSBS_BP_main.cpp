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
std::mt19937 rs_gen{0};

const std::array<std::string, 10> stripedVideoIdName = {
    {"0000", "0004", "0005", "0006", "0008", "0014", "0017", "0022", "0041", "0046"}};
const std::array<std::string, 2> invertVideoIdName = {{"0009", "0027"}};
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
Mat hstrech( const Mat& img )
{
   Mat simg( img.rows, 2 * img.cols, CV_32FC3 );

#pragma omp parallel for
   for ( size_t y = 0; y < simg.rows; y++ )
   {
      vec3* simg_data = simg.ptr<vec3>( y );

      for ( size_t x = 0; x < simg.cols; x++ )
      {
         simg_data[x] = cv_utils::imsample32FC3<vec3>( img, vec2( 0.5 * x, y ) );
      }
   }

   return simg;
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
bool processDepth( Mat& depth, const Mat& img, const Mat& undef, unsigned nIt )
{
   Mat curr( depth.rows, depth.cols, CV_32FC4 );
   // fill the first level
#pragma omp parallel for
   for ( unsigned y = 0; y < depth.rows; y++ )
   {
      const float* depthPtr = depth.ptr<float>( y );
      const float* undefPtr = undef.ptr<float>( y );
      const vec3* imgPtr = img.ptr<vec3>( y );
      vec4* currPtr = curr.ptr<vec4>( y );
      for ( unsigned x = 0; x < depth.cols; x++ )
      {
         const float& d = depthPtr[x];
         const vec3& color = imgPtr[x];
         const bool isUndef = undefPtr[x] < 0.5;

         currPtr[x] = vec4( isUndef ? -1.0 : d, color.x, color.y, color.z );
      }
   }
   Mat next( depth.rows, depth.cols, CV_32FC4 );

   unsigned nUndef = 0;
   for ( unsigned it = 0; it < nIt; ++it )
   {
      nUndef = 0;

#pragma omp parallel for
      for ( unsigned y = 0; y < depth.rows; y++ )
      {
         const vec4* currPtr = curr.ptr<vec4>( y );
         vec4* nextPtr = next.ptr<vec4>( y );

         for ( unsigned x = 0; x < depth.cols; x++ )
         {
            const vec4& currVal = currPtr[x];
            nextPtr[x] = currVal;

            if ( currVal.x < 0.0 )
            {
#pragma omp atomic
               nUndef++;
               float w = 0.0;
               float d = 0.0;
               for ( int dy = -1; dy <= 1; dy++ )
               {
                  for ( int dx = -1; dx <= 1; dx++ )
                  {
                     const vec2 dpos( (float)x + (float)dx, (float)y + (float)dy );
                     const vec4 dval = cv_utils::imsample32F<vec4>( curr, dpos );

                     if ( dval.x > 0.0 )
                     {
                        float lw = ( currVal.y - dval.y ) * ( currVal.y - dval.y ) +
                                   ( currVal.z - dval.z ) * ( currVal.z - dval.z ) +
                                   ( currVal.w - dval.w ) * ( currVal.w - dval.w );
                        lw = exp( -lw / 0.01 );
                        w += lw;
                        d += lw * dval.x;
                     }
                  }
               }
               if ( w > 0.0 )
               {
                  nextPtr[x].x = d / w;
               }
            }
         }
      }
      if ( nUndef == 0 ) break;
      std::swap( curr, next );
   }

#pragma omp parallel for
   for ( unsigned y = 0; y < depth.rows; y++ )
   {
      const vec4* currPtr = curr.ptr<vec4>( y );
      float* depthPtr = depth.ptr<float>( y );
      for ( unsigned x = 0; x < depth.cols; x++ )
      {
         depthPtr[x] = currPtr[x].x;
      }
   }

   return nUndef == 0;
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
                            ( err_data[x] > 0.31 ) || ( distance( mtR, mtRL ) > 0.75 );

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
    "{@imgFileLst      |         | images list   }"
    "{@imgRootDir    |         | images root dir   }"
    "{@imgOutDir    |         | images output dir   }"
    "{startIdx    |0         |    }"
    "{show     |         |    }"
    "{nowrite     |        |    }";

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
   const int startIdx = parser.get<int>( "startIdx" );

   const bool doShow = parser.get<bool>( "show" );
   const bool doWrite = !parser.get<bool>( "nowrite" );

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
   OclVarOpticalFlow ofEstimator( maxSz.x, maxSz.y, false, ofParams );

   // left / right random swap
   uniform_int_distribution<> rs_left( 0, 1 );

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

         const filesystem::path outVideoPath( videoIdname );

         const bool ignore =
             find( ignoreVideoIdName.begin(), ignoreVideoIdName.end(), videoIdname ) !=
             ignoreVideoIdName.end();
         if ( ignore ) continue;

         Mat img = cv_utils::imread32FC3( data[j], toLinear, true );

         const bool isStriped =
             find( stripedVideoIdName.begin(), stripedVideoIdName.end(), videoIdname ) !=
             stripedVideoIdName.end();

         const bool isInverted =
             find( invertVideoIdName.begin(), invertVideoIdName.end(), videoIdname ) !=
             invertVideoIdName.end();

         // split the current image
         Mat right, left;
         split_hsbs( img, right, left, isStriped ? ( img.rows - img.rows / 1.35 ) / 2 : 0 );
         // randomly swap left / right to avoid a potential right bias
         const bool doLeft = false;  // rs_left( rs_gen );
         if ( doLeft ) swap( right, left );

         right = hstrech( right );
         resizeToMin( right, maxSz );
         left = hstrech( left );
         resizeToMin( left, maxSz );

         // clone the image for output
         Mat oright = right.clone();

         // apply a blur to ease the optical flow estimation
         GaussianBlur( right, right, Size( 3, 3 ), 0.13 );
         GaussianBlur( left, left, Size( 3, 3 ), 0.13 );

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
         if ( cv::mean( cv::abs( ofRight ) )[0] < 1.0 ) continue;

         // compute the left to right optical flow
         Mat ofLeft( right.rows, right.cols, CV_32FC2 );
         ofEstimator.compute(
             reinterpret_cast<const float*>( right.ptr() ),
             reinterpret_cast<const float*>( left.ptr() ),
             right.cols,
             right.rows,
             reinterpret_cast<float*>( ofLeft.ptr() ) );

         // filter on low disparities
         if ( cv::mean( cv::abs( ofLeft ) )[0] < 1.0 ) continue;

         // compute the disparity and disparity mask
         Mat depth, mask;
         if ( !flowToDisp( ofRight, ofLeft, right, left, depth, mask, isInverted ^ doLeft ) )
            continue;

         cvtColor( oright, oright, COLOR_BGR2RGB );

         const filesystem::path fRight(
             videoIdname + "/" + outBasename + string( "_rgb" ) + ".png" );
         const filesystem::path fDepth( videoIdname + "/" + outBasename + string( "_d" ) + ".exr" );
         const filesystem::path fMask( videoIdname + "/" + outBasename + string( "_a" ) + ".png" );

         if ( doWrite )
         {
            imwrite( filesystem::path( outRootPath / fRight ).string().c_str(), oright * 255.0 );
            imwrite( filesystem::path( outRootPath / fDepth ).string().c_str(), depth );
            imwrite( filesystem::path( outRootPath / fMask ).string().c_str(), mask );
         }

         cout << fRight.string() << " " << fDepth.string() << " " << fMask.string();

         // if ( j == 2 )
         cout << endl;
         /*else
            cout << " ";*/

         // display
         if ( doShow )
         {
            // cvtColor( img, img, COLOR_BGR2RGB );
            // imshow( "Full", img );
            imshow( "Right", oright );
            cvtColor( left, left, COLOR_BGR2RGB );
            imshow( "Left", left );
            imshow( "Disp", depth );
            imshow( "Mask", mask );
            waitKey( 0 );
         }
      }
   }

   return ( 0 );
}
