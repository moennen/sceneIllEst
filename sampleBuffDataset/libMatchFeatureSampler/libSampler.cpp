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

#include <glm/glm.hpp>

#include <future>

#include <iostream>
#include <array>

using namespace std;
using namespace cv;
using namespace glm;

namespace
{

/*
 * The sampler takes as input 3 buffers 
 * 2 RGB images buffers : I_0 and I_1
 * 1 UV matching buffer : UV_M = UV([0:1]) st one search to minimize || I_1[ UV_M ] - I_0 || 
 *                        UV_N = UV([2:3]) st one search to maximize || I_1[ UV_N ] - I_0 || 
 * --> NB : the UV buffer should not be considered in == (0,0)
 * 
 * it return all 3 buffers : 
 * 
 **/

struct Sampler final
{
   mt19937 _rng;
   uniform_int_distribution<> _dataGen;
   uniform_real_distribution<> _tsGen;
   normal_distribution<> _rnGen;

   const ivec3 _sampleSz;
   const bool _toLinear;
   const bool _doRescale;
   
   const unsigned _fullBuffSz;
   future<bool> _asyncSample;
   vector<float> _asyncBuff;

   enum
   {
      nInBuffers = 3,
      nOutBuffers = 3,
      nOutPlanes = 10  // I_0 = 3 + I_1 = 3 + UV = 4
   };
   ImgNFileLst _data;

   inline static unsigned getBufferDepth( const unsigned buffId ) { return buffId > 1 ? 4 : 3; }

   Sampler(
       const char* dataSetPath,
       const char* dataPath,
       const ivec3 sampleSz,
       const bool toLinear,
       const bool doRescale,
       const bool doAsync,
       const int seed )
       : _rng( seed ),
         _tsGen( 0.0, 1.0 ),
         _sampleSz( sampleSz ),
         _toLinear( toLinear ),
         _doRescale( doRescale ),
         _fullBuffSz( _sampleSz.y * _sampleSz.z * nOutPlanes ),
         _data( nInBuffers )
   {
      HOP_PROF_FUNC();

      _data.open( dataSetPath, dataPath );

      if ( _data.size() )
      {
         cout << "Read dataset " << dataSetPath << " (" << _data.size() << ") " << endl;
      }
      _dataGen = uniform_int_distribution<>( 0, _data.size() - 1 );

      if ( _data.size() && doAsync )
      {
         _asyncBuff.resize( sampleSz.x * _fullBuffSz );
         _asyncSample = async( launch::async, [&]() { return sample_internal( &_asyncBuff[0] ); } );
      }
   }

   bool sample( float* buff )
   {
      if ( _asyncBuff.empty() )
         return sample_internal( buff );
      else
      {
         bool success = _asyncSample.get();
         if ( success )
         {
#pragma omp parallel for
            for ( int b = 0; b < _sampleSz.x; ++b )
            {
               const size_t off = b * _fullBuffSz;
               memcpy( buff + off, &_asyncBuff[off], sizeof( float ) * _fullBuffSz );
            }
         }
         _asyncSample = async( launch::async, [&]() { return sample_internal( &_asyncBuff[0] ); } );
         return success;
      }
   }

   size_t nSamples() const { return _data.size(); }
   ivec3 sampleSizes() const { return _sampleSz; }

  private:
   bool sample_internal( float* buff )
   {
      HOP_PROF_FUNC();

      const size_t szBuffC3 = _sampleSz.y * _sampleSz.z * 3;
      const size_t szBuffC2 = _sampleSz.y * _sampleSz.z * 2 ;

      const size_t szBuffOffI_1 = _sampleSz.x * szBuffC3;
      const size_t szBuffOffUV = szBuffOffI_1 + _sampleSz.x * szBuffC3;
      
      std::vector<char> sampled( _sampleSz.x, 0 );
      std::vector<size_t> v_si( _sampleSz.x );
      do
      {
         for ( auto& si : v_si ) si = _dataGen( _rng );
#pragma omp parallel for
         for ( size_t s = 0; s < _sampleSz.x; ++s )
         {
            if ( sampled[s] ) continue;

            const size_t si = v_si[s];

            Mat Img_0 = cv_utils::imread32FC3( _data.filePath( si, 0 ), _toLinear, true /*toRGB*/ );
            Mat Img_1 = cv_utils::imread32FC3( _data.filePath( si, 1 ), _toLinear, true /*toRGB*/ );
            Mat UV_M = cv_utils::imread32FC3( _data.filePath( si, 2 ), _toLinear, false /*toRGB*/ );
            Mat currLabels =
               cv_utils::imread32FC1( _data.filePath( si, 1 ), 1.0 /*Keep the labels as is*/ );

            ivec2 imgSz( currImg.cols, currImg.rows );

            // ignore too small samples
            if ( ( imgSz.x < _sampleSz.y ) || ( imgSz.y < _sampleSz.z ) ) continue;

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
                  // labels are resized in nearest because linear interpolation has no sense in a
                  // discrete scape
                  resize( currLabels, currLabels, Size(), ds, ds, INTER_NEAREST );
                  imgSz = ivec2( currImg.cols, currImg.rows );
               }
            }

            // random translate
            const ivec2 trans(
               std::floor( _tsGen( _rng ) * ( imgSz.x - _sampleSz.y ) ),
               std::floor( _tsGen( _rng ) * ( imgSz.y - _sampleSz.z ) ) );

            // crop
            currImg = currImg( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );
            currLabels = currLabels( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );

            // random small blur to remove artifacts + copy to destination
            Mat imgSple( _sampleSz.z, _sampleSz.y, CV_32FC3, buff + s * szBuffC3);
            cv_utils::adjustContrastBrightness<vec3>(
               currImg, ( 1.0f + 0.11f * _rnGen( _rng ) ), 0.11f * _rnGen( _rng ) );
            GaussianBlur( currImg, imgSple, Size( 5, 5 ), 0.31f * abs( _rnGen( _rng ) ) );
            Mat labelsSple( _sampleSz.z, _sampleSz.y, CV_32FC1, buff + szBuffOffLabels + s * szBuffC1 );
            Mat maskSple( _sampleSz.z, _sampleSz.y, CV_32FC1, buff + szBuffOffMask + s * szBuffC1 );
            // copy labels and apply a mapping if needed
            applyObjectMapping( currLabels, labelsSple, maskSple, _objectMapping );

            sampled[s] = 1;
         }
      } while ( accumulate( sampled.begin(), sampled.end(), 0 ) != _sampleSz.x );
      
      return true;
   }
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
      d[0] = sz.z;
      d[1] = sz.y;
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
   const ivec3 sz( params[0], params[2], params[1] );
   const bool toLinear( nParams > 3 ? params[3] > 0.0 : false );
   const bool doRescale( nParams > 4 ? params[4] > 0.0 : true );
   const int objectMapping( nParams > 5 ? static_cast<int>( params[5] ) : -1 );
   const bool doAsync( nParams > 6 ? static_cast<int>( params[6] ) : true );
   g_samplers[sidx].reset( new Sampler(
       datasetPath, dataPath, sz, toLinear, doRescale, objectMapping, doAsync, seed ) );

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
