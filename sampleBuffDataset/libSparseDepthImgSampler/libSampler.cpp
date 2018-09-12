/*! *****************************************************************************
 *   \file libSampler.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-04-06
 *   *****************************************************************************/

#include "sampleBuffDataset/libBuffDatasetSampler.h"

#include "utils/cv_utils.h"
#include "utils/imgFileLst.h"
#include "utils/Hop.h"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <glm/glm.hpp>

#include <random>
#include <fstream>
#include <string>
#include <vector>
#include <array>

#include <ctime>
#include <iostream>
#include <map>

using namespace std;
using namespace cv;
using namespace glm;

#define TRACE cout << __PRETTY_FUNCTION__ << "@" << __LINE__ << endl;

namespace
{
// copy the input mask to the output and set null the averaged mask values
void correctMask( const Mat& imask, Mat& omask )
{
#pragma omp parallel for
   for ( unsigned y = 0; y < omask.rows; y++ )
   {
      const float* imaskPtr = imask.ptr<float>( y );
      float* omaskPtr = omask.ptr<float>( y );
      for ( unsigned x = 0; x < omask.cols; x++ )
      {
         omaskPtr[x] = ( imaskPtr[x] < 0.999 ? 0.0 : 1.0 );
      }
   }
}

struct Sampler final
{
   mt19937 _rng;
   uniform_int_distribution<> _dataGen;
   uniform_real_distribution<> _tsGen;
   normal_distribution<> _rnGen;

   const ivec3 _sampleSz;
   const bool _toLinear;
   const bool _doRescale;

   const unsigned _depthBuffSz;
   const unsigned _maskBuffSz;
   const unsigned _imgBuffSz;

   enum
   {
      nInBuffers = 3,
      nOutBuffers = 4
   };
   ImgNFileLst _data;

   inline static unsigned getBufferDepth( const unsigned buffId ) { return buffId == 0 ? 3 : 1; }

   Sampler(
       const char* dataSetPath,
       const char* dataPath,
       const ivec3 sampleSz,
       const bool toLinear,
       const bool doRescale,
       const int seed )
       : _rng( seed ),
         _tsGen( 0.0, 1.0 ),
         _sampleSz( sampleSz ),
         _toLinear( toLinear ),
         _doRescale( doRescale ),
         _depthBuffSz( _sampleSz.y * _sampleSz.z ),
         _maskBuffSz( _depthBuffSz ),
         _imgBuffSz( _depthBuffSz * 3 ),
         _data( nInBuffers )
   {
      HOP_PROF_FUNC();

      _data.open( dataSetPath, dataPath );

      if ( _data.size() )
      {
         std::cout << "Read dataset " << dataSetPath << " (" << _data.size() << ") " << std::endl;
      }
      _dataGen = uniform_int_distribution<>( 0, _data.size() - 1 );

      cout << endl << "WARNING ! SAMPLER : NO ERODED MASK PRODUCED !!!!!!! " << endl << endl;
   }

   bool sample( float* buff )
   {
      HOP_PROF_FUNC();

      float* currBuffImg = buff;
      float* currBuffDepth = buff + _imgBuffSz * _sampleSz.x;
      float* currBuffMask = buff + ( _imgBuffSz + _depthBuffSz ) * _sampleSz.x;
      float* currBuffErodedMask = buff + ( _imgBuffSz + _depthBuffSz + _maskBuffSz ) * _sampleSz.x;

      for ( size_t s = 0; s < _sampleSz.x; ++s )
      {
         const size_t si = _dataGen( _rng );

         Mat currImg = cv_utils::imread32FC3( _data.filePath( si, 0 ), _toLinear, true /*toRGB*/ );
         Mat currDepth = cv_utils::imread32FC1( _data.filePath( si, 1 ) );
         Mat currMask = cv_utils::imread32FC1( _data.filePath( si, 2 ) );

         ivec2 imgSz( currImg.cols, currImg.rows );

         // ignore failed samples
         if ( currImg.empty() || currDepth.empty() || currMask.empty() )
         {
            --s;
            continue;
         }

         // padd too small samples
         if ( ( imgSz.x < _sampleSz.y ) || ( imgSz.y < _sampleSz.z ) )
         {
            const ivec2 topright(
                max( ivec2( _sampleSz.y - imgSz.x, _sampleSz.z - imgSz.y ), ivec2( 0 ) ) );
            copyMakeBorder( currImg, currImg, topright.y, 0, 0, topright.x, BORDER_CONSTANT );
            copyMakeBorder( currDepth, currDepth, topright.y, 0, 0, topright.x, BORDER_CONSTANT );
            copyMakeBorder( currMask, currMask, topright.y, 0, 0, topright.x, BORDER_CONSTANT );
            imgSz = ivec2( currImg.cols, currImg.rows );
         }

         // random rescale
         bool rescaled = false;
         if ( _doRescale )
         {
            const float minDs =
                std::max( (float)_sampleSz.z / imgSz.y, (float)_sampleSz.y / imgSz.x );
            const float ds = mix( 1.0f, minDs, _tsGen( _rng ) );
            if ( ds < 1.0 )
            {
               rescaled = true;
               resize( currImg, currImg, Size(), ds, ds, INTER_AREA );
               resize( currDepth, currDepth, Size(), ds, ds, INTER_AREA );
               resize( currMask, currMask, Size(), ds, ds, INTER_AREA );
               imgSz = ivec2( currImg.cols, currImg.rows );
            }
         }

         // random translate
         const ivec2 trans(
             std::floor( _tsGen( _rng ) * ( imgSz.x - _sampleSz.y ) ),
             std::floor( _tsGen( _rng ) * ( imgSz.y - _sampleSz.z ) ) );

         // crop
         currImg = currImg( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );
         currDepth = currDepth( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );
         currMask = currMask( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );

         // copy and correct mask
         Mat maskSple( _sampleSz.z, _sampleSz.y, CV_32FC1, currBuffMask );
         if ( rescaled )
            correctMask( currMask, maskSple );
         else
            currMask.copyTo( maskSple );
         if (sum(maskSple)[0] < 20)
         {
            --s;
            continue;
         }

         // random small blur to remove artifacts + copy to destination
         Mat imgSple( _sampleSz.z, _sampleSz.y, CV_32FC3, currBuffImg );
         cv_utils::adjustContrastBrightness<vec3>(
             currImg, ( 1.0f + 0.11f * _rnGen( _rng ) ), 0.11f * _rnGen( _rng ) );
         GaussianBlur( currImg, imgSple, Size( 5, 5 ), 0.31 * abs( _rnGen( _rng ) ) );

         // copy and process depth
         Mat depthSple( _sampleSz.z, _sampleSz.y, CV_32FC1, currBuffDepth );
         maskSple.convertTo( currMask, CV_8UC1, 255.0, 0.0 );
         cv::Mat mean, std;
         cv::meanStdDev( currDepth, mean, std, currMask );
         depthSple = (( currDepth - mean ) / std);
         //normalize( currDepth, depthSple, 0.0, 1.0, NORM_MINMAX, -1, currMask );
         // this is for debugging !!!
         depthSple = depthSple.mul( maskSple );

         // copy and process eroded mask
         // Mat erodedMaskSple( _sampleSz.z, _sampleSz.y, CV_32FC1, currBuffErodedMask );
         // erode(maskSple, erodedMaskSple, Mat());

         currBuffImg += _imgBuffSz;
         currBuffDepth += _depthBuffSz;
         currBuffMask += _maskBuffSz;
         currBuffErodedMask += _maskBuffSz;
      }

      return true;
   }

   size_t nSamples() const { return _data.size(); }
   ivec3 sampleSizes() const { return _sampleSz; }
};

array<unique_ptr<Sampler>, 33> g_samplers;
};

extern "C" int getNbBuffers( const int /*sidx*/ ) { return Sampler::nOutBuffers; }

extern "C" int getBuffersDim( const int sidx, float* dims )
{
   HOP_PROF_FUNC();

   if ( !g_samplers[sidx].get() ) return ERROR_UNINIT;

   const ivec3 sz = g_samplers[sidx]->sampleSizes();
   float* d = dims;
   for ( size_t i = 0; i < Sampler::nOutBuffers; ++i )
   {
      d[0] = sz.y;
      d[1] = sz.z;
      d[2] = Sampler::getBufferDepth( i );
      d += 3;
   }

   return SUCCESS;
}

extern "C" int initBuffersDataSampler(
    const int sidx,
    const char* datasetPath,
    const char* dataPath,
    const int nParams,
    const float* params,
    const int seed )
{
   HOP_PROF_FUNC();

   // check input
   if ( ( nParams < 3 ) || ( sidx > g_samplers.size() ) ) return ERROR_BAD_ARGS;

   // parse params
   const ivec3 sz( params[0], params[1], params[2] );
   const bool toLinear( nParams > 3 ? params[3] > 0.0 : false );
   const bool doRescale( nParams > 4 ? params[4] > 0.0 : false );
   g_samplers[sidx].reset( new Sampler( datasetPath, dataPath, sz, toLinear, doRescale, seed ) );

   return g_samplers[sidx]->nSamples() ? SUCCESS : ERROR_BAD_DB;
}

extern "C" int getBuffersDataSample( const int sidx, float* buff )
{
   HOP_PROF_FUNC();

   if ( !g_samplers[sidx].get() ) return ERROR_UNINIT;

   // sample
   if ( !g_samplers[sidx]->sample( buff ) ) return ERROR_GENERIC;

   return SUCCESS;
}
