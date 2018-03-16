/*! *****************************************************************************
 *   \file cv_utils.C
 *   \author moennen
 *   \brief
 *   \date 2018-03-16
 *   *****************************************************************************/
#include <utils/cv_utils.h>

namespace
{
inline std::array<float, 256> generate8UCToLinear32FLUT()
{
   std::array<float, 256> lut;
   for ( size_t b = 0; b < 256; ++b )
   {
      lut[b] = cv_utils::toLinear( b / 255.0f );
   }
   return lut;
}
}

cv::Mat cv_utils::convert8UC3ToLinear32FC3( cv::Mat& img )
{
   static const std::array<float, 256> ucToLinear32f = generate8UCToLinear32FLUT();

   HOP_PROF_FUNC();
   const size_t rowSz = img.cols * 3;
   cv::Mat conv( img.rows, img.cols, CV_32FC3 );
#pragma omp parallel for
   for ( size_t y = 0; y < img.rows; y++ )
   {
      const unsigned char* row_idata = img.ptr<unsigned char>( y );
      float* row_odata = conv.ptr<float>( y );
      for ( size_t x = 0; x < rowSz; x++ )
      {
         row_odata[x] = ucToLinear32f[row_idata[x]];
      }
   }
   return conv;
}