//*****************************************************************************/
//
// Filename cv_utils.h
//
// Copyright (c) 2017 Autodesk, Inc.
// All rights reserved.
//
// This computer source code and related instructions and comments are the
// unpublished confidential and proprietary information of Autodesk, Inc.
// and are protected under applicable copyright and trade secret law.
// They may not be disclosed to, copied or used by any third party without
// the prior written consent of Autodesk, Inc.
//*****************************************************************************/
#ifndef _UTILS_CV_UTILS_H
#define _UTILS_CV_UTILS_H

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <glm/glm.hpp>

#include <iostream>

namespace cv_utils
{
inline cv::Mat imread32FC3( const std::string& imgPath )
{
   cv::Mat img = cv::imread( imgPath, cv::IMREAD_UNCHANGED );
   if ( !img.data || ( img.channels() < 3 ) || ( img.channels() > 4 ) )
   {
      std::cerr << "ERROR loading image : " << imgPath << std::endl;
      return cv::Mat();
   }
   if ( img.channels() == 4 ) cv::cvtColor( img, img, cv::COLOR_RGBA2RGB );
   if ( img.type() != CV_32F )
   {
      img.convertTo( img, CV_32F );
      img /= 255.0;
   }
   return img;
}

inline cv::Vec3f imsample32FC3( const cv::Mat& img, glm::vec2 in_pt )
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

   return cv::Vec3f( bgr.x, bgr.y, bgr.z );
}
}

#endif  // _UTILS_CV_UTILS_H
