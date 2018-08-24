/*! *****************************************************************************
 *   \file cv_utils.h
 *   \author moennen
 *   \brief
 *   \date 2018-03-16
 *   *****************************************************************************/
#ifndef _UTILS_CV_UTILS_H
#define _UTILS_CV_UTILS_H

#include "utils/Hop.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <glm/glm.hpp>

#include <iostream>

namespace cv_utils
{
inline float toLinear( float c )
{
   return c <= 0.04045f ? c / 12.92f : std::pow( ( c + 0.055f ) / 1.055f, 2.4f );
}

inline float toLog( float c )
{
   return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow( c, 1.0f / 2.4f ) - 0.055f;
}

inline void normalizeMeanStd( cv::Mat& img )
{
   cv::Mat mean, std;
   cv::meanStdDev( img, mean, std );
   img = ( img - mean ) / std;
}

inline void imToLinear( cv::Mat& img )
{
   HOP_PROF_FUNC();
   const size_t rowSz = img.cols * 3;
#pragma omp parallel for
   for ( size_t y = 0; y < img.rows; y++ )
   {
      float* row_data = img.ptr<float>( y );
      for ( size_t x = 0; x < rowSz; x++ )
      {
         row_data[x] = toLinear( row_data[x] );
      }
   }
}

inline void imToLog( cv::Mat& img )
{
   const size_t rowSz = img.cols * 3;
#pragma omp parallel for
   for ( size_t y = 0; y < img.rows; y++ )
   {
      float* row_data = img.ptr<float>( y );
      for ( size_t x = 0; x < rowSz; x++ )
      {
         row_data[x] = toLog( row_data[x] );
      }
   }
}

inline cv::Mat imread32FC1( const std::string& imgPath )
{
   HOP_PROF_FUNC();

   cv::Mat img;
   {
      HOP_PROF( "cv_imread_c1" );
      img = cv::imread( imgPath, cv::IMREAD_UNCHANGED );
   }
   if ( !img.data || ( img.channels() > 1 ) )
   {
      std::cerr << "ERROR loading c1 image  : " << imgPath << std::endl;
      return cv::Mat();
   }
   if ( img.type() != CV_32F )
   {
      HOP_PROF( "cv_convert" );
      img.convertTo( img, CV_32F );
   }
   return img;
}

cv::Mat convert8UC3ToLinear32FC3( cv::Mat& img );

inline cv::Mat imread32FC3(
    const std::string& imgPath,
    bool toLinear = false,
    bool toRGB = false,
    const float smax = 255.0 )
{
   HOP_PROF_FUNC();

   cv::Mat img;
   {
      HOP_PROF( "cv_imread" );
      img = cv::imread( imgPath, cv::IMREAD_UNCHANGED );
   }
   if ( !img.data || ( img.channels() == 2 ) || ( img.channels() > 4 ) )
   {
      std::cerr << "ERROR loading image : " << imgPath << std::endl;
      return cv::Mat();
   }
   if ( img.channels() == 1 )
      cv::cvtColor( img, img, toRGB ? cv::COLOR_GRAY2BGR : cv::COLOR_GRAY2RGB );
   else if ( img.channels() == 4 )
      cv::cvtColor( img, img, toRGB ? cv::COLOR_RGBA2BGR : cv::COLOR_RGBA2RGB );
   else if ( toRGB )
      cv::cvtColor( img, img, cv::COLOR_BGR2RGB );
   if ( ( img.type() == CV_8UC3 ) && toLinear )
   {
      img = convert8UC3ToLinear32FC3( img );
   }
   else
   {
      if ( img.depth() != CV_32F )
      {
         HOP_PROF( "cv_convert" );
         img.convertTo( img, CV_32F );
         img /= smax;
      }
      if ( toLinear ) imToLinear( img );
   }
   return img;
}

inline cv::Mat imread32FC4( const std::string& imgPath, bool toLinear = false, bool toRGB = true )
{
   cv::Mat img = imread32FC3( imgPath, toLinear );
   if ( !img.empty() ) cv::cvtColor( img, img, toRGB ? cv::COLOR_BGR2RGBA : cv::COLOR_RGB2RGBA );
   return img;
}

template <class TVec3>
inline TVec3 imsample32FC3( const cv::Mat& img, const glm::vec2& in_pt )
{
   // compute the positions
   const glm::vec2 max_pt( img.cols - 1, img.rows - 1 );
   const glm::vec2 pt = glm::clamp( in_pt, glm::vec2( 0.0 ), max_pt );
   const glm::ivec2 ul_pt(
       static_cast<int>( std::floor( pt.x ) ), static_cast<int>( std::floor( pt.y ) ) );

   // fetch the data
   const float* u_row = img.ptr<float>( ul_pt.y ) + 3 * ul_pt.x;
   const glm::vec3 ul( u_row[0], u_row[1], u_row[2] );
   const glm::vec3 ur = ul_pt.x < max_pt.x ? glm::vec3( u_row[3], u_row[4], u_row[5] ) : ul;
   const float* b_row = ul_pt.y < max_pt.y ? img.ptr<float>( ul_pt.y + 1 ) + 3 * ul_pt.x : u_row;
   const glm::vec3 bl( b_row[0], b_row[1], b_row[2] );
   const glm::vec3 br = ul_pt.x < max_pt.x ? glm::vec3( b_row[3], b_row[4], b_row[5] ) : bl;

   // linear interpolation
   const glm::vec3 bgr = glm::mix(
       glm::mix( ul, ur, pt.x - ul_pt.x ), glm::mix( bl, br, pt.x - ul_pt.x ), pt.y - ul_pt.y );

   return TVec3( bgr.x, bgr.y, bgr.z );
}

template <class TVec>
inline TVec imsample32F( const cv::Mat& img, const glm::vec2& in_pt )
{
   // compute the positions
   const glm::vec2 max_pt( img.cols - 1, img.rows - 1 );
   const glm::vec2 pt = glm::clamp( in_pt, glm::vec2( 0.0 ), max_pt );
   const glm::ivec2 ul_pt(
       static_cast<int>( std::floor( pt.x ) ), static_cast<int>( std::floor( pt.y ) ) );

   // fetch the data
   const TVec& ul = img.ptr<TVec>( ul_pt.y )[ul_pt.x];
   const TVec& ur = ul_pt.x < max_pt.x ? img.ptr<TVec>( ul_pt.y )[ul_pt.x + 1] : ul;

   const TVec& bl = ul_pt.y < max_pt.y ? img.ptr<TVec>( ul_pt.y + 1 )[ul_pt.x] : ul;
   const TVec& br = ul_pt.y < max_pt.y
                        ? ( ul_pt.x < max_pt.x ? img.ptr<TVec>( ul_pt.y + 1 )[ul_pt.x + 1] : bl )
                        : ur;

   // linear interpolation
   return glm::mix(
       glm::mix( ul, ur, pt.x - ul_pt.x ), glm::mix( bl, br, pt.x - ul_pt.x ), pt.y - ul_pt.y );
}

inline void imToBuffer( const cv::Mat& img, float* buff, const bool toRGB = false )
{
   HOP_PROF_FUNC();
   const size_t row_stride = img.cols * img.channels();
   const size_t row_size = sizeof( float ) * row_stride;

#pragma omp parallel for
   for ( size_t y = 0; y < img.rows; y++ )
   {
      const float* row_img_data = img.ptr<float>( y );
      float* row_buff_data = buff + y * row_stride;
      memcpy( row_buff_data, row_img_data, row_size );
   }
}

inline void fittResizeCrop( cv::Mat& img, const glm::uvec2 sampleSz )
{
   glm::uvec2 imgSz( img.cols, img.rows );

   // rescale
   const float ds = std::min( (float)sampleSz.y / imgSz.y, (float)sampleSz.x / imgSz.x );
   cv::resize( img, img, cv::Size(), ds, ds, CV_INTER_AREA );
   imgSz = glm::uvec2( img.cols, img.rows );

   // translate
   const glm::ivec2 trans(
       std::floor( 0.5 * ( imgSz.x - sampleSz.x ) ), std::floor( 0.5 * ( imgSz.y - sampleSz.y ) ) );

   // crop
   img = img( cv::Rect( trans.x, trans.y, sampleSz.x, sampleSz.y ) ).clone();
}
}

#endif  // _UTILS_CV_UTILS_H
