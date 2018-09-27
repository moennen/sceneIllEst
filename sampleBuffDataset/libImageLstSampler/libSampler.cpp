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

#include <random>
#include <array>
#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;

namespace
{
struct Sampler final
{
   mt19937 _rng;
   uniform_int_distribution<> _dataGen;
   uniform_real_distribution<> _tsGen;
   normal_distribution<> _rnGen;

   const int _nBuffers;

   const ivec3 _sampleSz;
   const bool _toLinear;
   const bool _doRescale;

   const unsigned _fullBuffSz;

   future<bool> _asyncSample;
   vector<float> _asyncBuff;

   ImgNFileLst _data;

   inline unsigned getNBuffers() { return _nBuffers; }
   inline static unsigned getBufferDepth( const unsigned buffId ) { return 3; }

   Sampler(
       const char* dataSetPath,
       const char* dataPath,
       const int nBuffers,
       const ivec3 sampleSz,
       const bool toLinear,
       const bool doRescale,
       const bool doAsync,
       const int seed )
       : _rng( seed ),
         _tsGen( 0.0, 1.0 ),
         _nBuffers( nBuffers ),
         _sampleSz( sampleSz ),
         _toLinear( toLinear ),
         _doRescale( doRescale ),
         _fullBuffSz( _sampleSz.y * _sampleSz.z * _nBuffers * 3 ),
         _data( nBuffers )
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

      const size_t szBuff = _sampleSz.y * _sampleSz.z * 3  ;
      const size_t szBatchBuffOff =  _sampleSz.x * szBuff;

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

            bool success = true;

            const float fRescaleFactor = _doRescale ? _tsGen( _rng ) : 0.0;
            const vec2 v2TransOffset( _tsGen( _rng ), _tsGen( _rng ) );

            for ( size_t ss = 0; ss < _nBuffers; ++ss )
            {
               Mat currImg =
                   cv_utils::imread32FC3( _data.filePath( si, ss ), _toLinear, true /*toRGB*/ );

               // ignore failed samples
               if ( currImg.empty() )
               {
                  success = false;
                  break;
               }

               ivec2 imgSz( currImg.cols, currImg.rows );

               // padd too small samples
               if ( ( imgSz.x < _sampleSz.y ) || ( imgSz.y < _sampleSz.z ) )
               {
                  const ivec2 topright(
                      max( ivec2( _sampleSz.y - imgSz.x, _sampleSz.z - imgSz.y ), ivec2( 0 ) ) );
                  copyMakeBorder( currImg, currImg, topright.y, 0, 0, topright.x, BORDER_CONSTANT );
                  imgSz = ivec2( currImg.cols, currImg.rows );
               }

               // random rescale
               if ( _doRescale )
               {
                  const float minDs =
                      std::max( (float)_sampleSz.z / imgSz.y, (float)_sampleSz.y / imgSz.x );
                  const float ds = mix( std::min( 1.0f, minDs ), minDs, fRescaleFactor );
                  resize( currImg, currImg, Size(), ds, ds, CV_INTER_AREA );
                  imgSz = ivec2( currImg.cols, currImg.rows );
               }

               // random translate
               const ivec2 trans(
                   std::floor( v2TransOffset.x * ( imgSz.x - _sampleSz.y ) ),
                   std::floor( v2TransOffset.y * ( imgSz.y - _sampleSz.z ) ) );

               // crop
               currImg = currImg( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );

               // random small blur to remove artifacts + copy to destination
               Mat imgSple( _sampleSz.z, _sampleSz.y, CV_32FC3, buff + ss * szBatchBuffOff + s * szBuff );
               cv_utils::adjustContrastBrightness<vec3>(
                   currImg, ( 1.0f + 0.11f * _rnGen( _rng ) ), 0.11f * _rnGen( _rng ) );
               GaussianBlur( currImg, imgSple, Size( 5, 5 ), 0.31 * abs( _rnGen( _rng ) ) );
            }

            if ( success ) sampled[s] = 1;
         }
      } while ( accumulate( sampled.begin(), sampled.end(), 0 ) != _sampleSz.x );

      return true;
   }
};

array<unique_ptr<Sampler>, 33> g_samplers;
};

extern "C" int getNbBuffers( const int sidx )
{
   if ( !g_samplers[sidx].get() )
      return 0;
   else
      return g_samplers[sidx]->getNBuffers();
}

extern "C" int getBuffersDim( const int sidx, float* dims )
{
   HOP_PROF_FUNC();

   if ( !g_samplers[sidx].get() ) return ERROR_UNINIT;

   const int nBuffers = g_samplers[sidx]->getNBuffers();
   const ivec3 sz = g_samplers[sidx]->sampleSizes();
   float* d = dims;
   for ( size_t i = 0; i < nBuffers; ++i )
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
   if ( ( nParams < 4 ) || ( sidx > g_samplers.size() ) ) return ERROR_BAD_ARGS;

   // parse params
   const ivec3 sz( params[0], params[1], params[2] );
   const int nBuffers = std::max( 1, static_cast<int>( params[3] ) );
   const bool toLinear( nParams > 4 ? params[4] > 0.0 : false );
   const bool doRescale( nParams > 5 ? params[5] > 0.0 : true );
   const bool doAsync( nParams > 6 ? params[6] > 0.0 : true );
   g_samplers[sidx].reset(
       new Sampler( datasetPath, dataPath, nBuffers, sz, toLinear, doRescale, doAsync, seed ) );

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
