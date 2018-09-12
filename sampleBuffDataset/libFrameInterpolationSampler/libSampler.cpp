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
struct Sampler final
{
   mt19937 _rng;
   normal_distribution<> _rnGen;
   uniform_int_distribution<> _dataGen;
   uniform_real_distribution<> _transGen;
   uniform_real_distribution<> _ldAsBlendGen;

   ImgNFileLst _data;
   const ivec3 _sampleSz;
   const bool _toLinear;
   const float _downsample;
   const float _ldAsBlendFreq;
   const float _minPrevNextSqDiff;
   const float _maxPrevNextSqDiff;
   enum
   {
      DefaultMode = 0,
      PrevIsClosestMode = 1
   };
   const int _mode;

   Sampler(
       const char* dataSetPath,
       const char* dataPath,
       const float downsampleFactor,
       const float ldAsBlendFreq,
       const float minPrevNextSqDiff,
       const float maxPrevNextSqDiff,
       const int mode,
       const ivec3 sampleSz,
       const bool toLinear,
       const int seed )
       : _rng( seed ),
         _transGen( 0.0, 1.0 ),
         _ldAsBlendGen( 0.0, 1.0 ),
         _data( 4 ),  // 3 files + 1 alpha
         _sampleSz( sampleSz ),
         _toLinear( toLinear ),
         _downsample( downsampleFactor ),
         _ldAsBlendFreq( ldAsBlendFreq ),
         _minPrevNextSqDiff( minPrevNextSqDiff ),
         _maxPrevNextSqDiff( maxPrevNextSqDiff ),
         _mode( mode )
   {
      HOP_PROF_FUNC();

      _data.open( dataSetPath, dataPath );
      if ( _data.size() )
      {
         std::cout << "Read dataset " << dataSetPath << " (" << _data.size() << ") "
                   << ldAsBlendFreq << " " << downsampleFactor << std::endl;
      }
      _dataGen = uniform_int_distribution<>( 0, _data.size() - 1 );
   }

   bool write() {}

   bool sample( float* buff )
   {
      HOP_PROF_FUNC();

      const unsigned buffSz = _sampleSz.z * _sampleSz.y * 3;
      const unsigned batchBuffSz = buffSz * _sampleSz.x;

      float* currBuffGTHD = buff;
      float* currBuffGTLD = buff + batchBuffSz;
      float* currBuffBlend = buff + 2 * batchBuffSz;
      float* currBuffPrev = buff + 3 * batchBuffSz;
      float* currBuffNext = buff + 4 * batchBuffSz;

      for ( size_t s = 0; s < _sampleSz.x; ++s )
      {
         const size_t si = _dataGen( _rng );

         float alpha = stof( _data( si, 3 ) );
         const bool swapPrevNext = ( alpha > 0.5 ) && ( _mode == PrevIsClosestMode );
         alpha = swapPrevNext ? 1.0 - alpha : alpha;

         const string pathA = _data.filePath( si, 0 );
         const string pathB = _data.filePath( si, 1 );
         const string pathC = _data.filePath( si, 2 );

         Mat prevImg = cv_utils::imread32FC3( swapPrevNext ? pathC : pathA, _toLinear );
         Mat currImg = cv_utils::imread32FC3( pathB, _toLinear );
         Mat nextImg = cv_utils::imread32FC3( swapPrevNext ? pathA : pathC, _toLinear );

         ivec2 imgSz( currImg.cols, currImg.rows );

         // ignore too small samples
         if ( ( imgSz.x < _sampleSz.y ) || ( imgSz.y < _sampleSz.z ) )
         {
            --s;
            continue;
         }

         // bad dataset : the 3 sample image have to be of the same size
         if ( ( prevImg.cols != imgSz.x ) || ( prevImg.rows != imgSz.y ) ||
              ( nextImg.cols != imgSz.x ) || ( nextImg.rows != imgSz.y ) )
            return false;

         // random rescale
         const float ds =
             mix( 1.0f,
                  std::max( (float)_sampleSz.z / imgSz.y, (float)_sampleSz.y / imgSz.x ),
                  _transGen( _rng ) );
         resize( prevImg, prevImg, Size(), ds, ds, INTER_AREA );
         resize( currImg, currImg, Size(), ds, ds, INTER_AREA );
         resize( nextImg, nextImg, Size(), ds, ds, INTER_AREA );
         imgSz = ivec2( currImg.cols, currImg.rows );

         // random translate
         const ivec2 trans(
             std::floor( _transGen( _rng ) * ( imgSz.x - _sampleSz.y ) ),
             std::floor( _transGen( _rng ) * ( imgSz.y - _sampleSz.z ) ) );

         // crop
         prevImg = prevImg( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );
         currImg = currImg( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );
         nextImg = nextImg( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );

         // preprocess
         const float contrast = ( 1.0f + 0.11f * _rnGen( _rng ) );
         const float brightness = 0.11f * _rnGen( _rng );
         cv_utils::adjustContrastBrightness<vec3>( prevImg, contrast, brightness );
         cv_utils::adjustContrastBrightness<vec3>( currImg, contrast, brightness );
         cv_utils::adjustContrastBrightness<vec3>( nextImg, contrast, brightness );
         // random small blur to remove artifacts
         const float blur = 1.5 * _transGen( _rng );
         GaussianBlur( prevImg, prevImg, Size( 5, 5 ), blur );
         GaussianBlur( currImg, currImg, Size( 5, 5 ), blur );
         GaussianBlur( nextImg, nextImg, Size( 5, 5 ), blur );

         // filter samples on differences :
         const double sqDiff = norm( prevImg, nextImg ) / buffSz;
         if ( ( sqDiff > _maxPrevNextSqDiff ) || ( sqDiff < _minPrevNextSqDiff ) )
         {
            --s;
            continue;
         }

         // For initial estimate we need to use the blend image as the LD
         // --> sample
         const bool blendInLD( _ldAsBlendGen( _rng ) < _ldAsBlendFreq );

         // currBuffGTHD / currBuffGTLD
         {
            Mat sple( _sampleSz.z, _sampleSz.y, CV_32FC3, currBuffGTHD );
            cvtColor( currImg, sple, COLOR_BGR2RGB );
            // downsample / upsample
            if ( !blendInLD )
            {
               Mat tmp;
               resize( sple, tmp, Size(), _downsample, _downsample, CV_INTER_AREA );
               sple = Mat( _sampleSz.z, _sampleSz.y, CV_32FC3, currBuffGTLD );
               resize( tmp, sple, Size( _sampleSz.y, _sampleSz.z ), 0, 0, CV_INTER_LINEAR );
            }
         }

         // prev / next
         {
            Mat sple( _sampleSz.z, _sampleSz.y, CV_32FC3, currBuffPrev );
            cvtColor( prevImg, sple, COLOR_BGR2RGB );
            sple = Mat( _sampleSz.z, _sampleSz.y, CV_32FC3, currBuffNext );
            cvtColor( nextImg, sple, COLOR_BGR2RGB );
         }

         // blend
         {
            Mat tmp = alpha * prevImg + ( 1.0 - alpha ) * nextImg;
            Mat sple( _sampleSz.z, _sampleSz.y, CV_32FC3, currBuffBlend );
            cvtColor( tmp, sple, COLOR_BGR2RGB );
            // downsample / upsample
            if ( blendInLD )
            {
               Mat tmp;
               resize( sple, tmp, Size(), _downsample, _downsample, CV_INTER_AREA );
               sple = Mat( _sampleSz.z, _sampleSz.y, CV_32FC3, currBuffGTLD );
               resize( tmp, sple, Size( _sampleSz.y, _sampleSz.z ), 0, 0, CV_INTER_LINEAR );
            }
         }

         currBuffGTHD += buffSz;
         currBuffGTLD += buffSz;
         currBuffBlend += buffSz;
         currBuffPrev += buffSz;
         currBuffNext += buffSz;
      }

      return true;
   }

   size_t nSamples() const { return _data.size(); }
   ivec3 sampleSizes() const { return _sampleSz; }
};

array<unique_ptr<Sampler>, 33> g_samplers;
};

extern "C" int getNbBuffers( const int /*sidx*/ ) { return 5; }

extern "C" int getBuffersDim( const int sidx, float* dims )
{
   HOP_PROF_FUNC();

   if ( !g_samplers[sidx].get() ) return ERROR_UNINIT;

   const ivec3 sz = g_samplers[sidx]->sampleSizes();
   float* d = dims;
   for ( size_t i = 0; i < 5; ++i )
   {
      d[0] = sz.y;
      d[1] = sz.z;
      d[2] = 3;
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
   if ( ( nParams < 8 ) || ( sidx > g_samplers.size() ) ) return ERROR_BAD_ARGS;

   // parse params
   const ivec3 sz( params[0], params[1], params[2] );
   const float downsampleFactor = params[3];
   const float ldAsBlendFreq = params[4];
   const float minPrevNextSqDiff = params[5];
   const float maxPrevNextSqDiff = params[6];
   const int mode = static_cast<int>( params[7] );
   const bool toLinear( nParams > 8 ? params[8] > 0.0 : false );
   g_samplers[sidx].reset( new Sampler(
       datasetPath,
       dataPath,
       downsampleFactor,
       ldAsBlendFreq,
       minPrevNextSqDiff,
       maxPrevNextSqDiff,
       mode,
       sz,
       toLinear,
       seed ) );

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
