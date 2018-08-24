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
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <glm/glm.hpp>

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
   boost::random::mt19937 _rng;
   boost::random::uniform_int_distribution<> _dataGen;
   boost::random::uniform_real_distribution<> _tsGen;

   static constexpr float maxDsScaleFactor = 3.5;

   const ivec3 _sampleSz;

   enum
   {
      nBuffers = 2
   };
   ImgNFileLst<nBuffers> _data;

   inline static unsigned getBufferDepth( const unsigned buffId ) { return buffId == 0 ? 3 : 1; }

   Sampler( const char* dataSetPath, const char* dataPath, const ivec3 sampleSz, const int seed )
       : _rng( seed ), _tsGen( 0.0, 1.0 ), _sampleSz( sampleSz )
   {
      HOP_PROF_FUNC();

      _data.open( dataSetPath, dataPath );

      if ( _data.size() )
      {
         std::cout << "Read dataset " << dataSetPath << " (" << _data.size() << ") " << std::endl;
      }
      _dataGen = boost::random::uniform_int_distribution<>( 0, _data.size() - 1 );
   }

   bool sample( float* buff )
   {
      HOP_PROF_FUNC();

      const unsigned depthBuffSz = _sampleSz.y * _sampleSz.z;
      const unsigned imgBuffSz = depthBuffSz * 3;

      float* currBuffImg = buff;
      float* currBuffDepth = buff + imgBuffSz * _sampleSz.x ;

      for ( size_t s = 0; s < _sampleSz.x; ++s )
      {
         const ImgNFileLst<nBuffers>::Data& data = _data[_dataGen( _rng )];

         Mat currImg = cv_utils::imread32FC3( data[0], false/*toLinear*/, true/*toRGB*/ );
         Mat currDepth = cv_utils::imread32FC1( data[1] );

         ivec2 imgSz( currImg.cols, currImg.rows );

         // ignore too small samples
         if ( ( imgSz.x < _sampleSz.y ) || ( imgSz.y < _sampleSz.z ) )
         {
            --s;
            continue;
         }

         // bad dataset : the samples o be of the same size
         if ( ( currDepth.cols != imgSz.x ) || ( currDepth.rows != imgSz.y ) ) return false;

         // random rescale
         const float minDs = std::max( (float)_sampleSz.z / imgSz.y, (float)_sampleSz.y / imgSz.x );
         const float ds = mix( std::min( 1.0f, maxDsScaleFactor * minDs ), minDs, _tsGen( _rng ) );
         resize( currImg, currImg, Size(), ds, ds, CV_INTER_AREA );
         resize( currDepth, currDepth, Size(), ds, ds, CV_INTER_AREA );
         imgSz = ivec2( currImg.cols, currImg.rows );

         // random translate
         const ivec2 trans(
             std::floor( _tsGen( _rng ) * ( imgSz.x - _sampleSz.y ) ),
             std::floor( _tsGen( _rng ) * ( imgSz.y - _sampleSz.z ) ) );

         // crop
         currImg = currImg( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );
         currDepth = currDepth( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );

         // random small blur to remove artifacts + copy to destination
         Mat imgSple( _sampleSz.z, _sampleSz.y, CV_32FC3, currBuffImg );
         GaussianBlur( currImg, imgSple, Size( 5, 5 ), 1.5 * _tsGen( _rng ) );
         Mat depthSple( _sampleSz.z, _sampleSz.y, CV_32FC1, currBuffDepth );
         GaussianBlur( currDepth, depthSple, Size( 5, 5 ), 1.5 * _tsGen( _rng ) );
         //cv_utils::normalizeMeanStd(depthSple);
         normalize( depthSple, depthSple, 0, 1, NORM_MINMAX );

         currBuffImg += imgBuffSz;
         currBuffDepth += depthBuffSz;
      }

      return true;
   }

   size_t nSamples() const { return _data.size(); }
   ivec3 sampleSizes() const { return _sampleSz; }
};

array<unique_ptr<Sampler>, 33> g_samplers;
};

extern "C" int getNbBuffers( const int /*sidx*/ ) { return Sampler::nBuffers; }

extern "C" int getBuffersDim( const int sidx, float* dims )
{
   HOP_PROF_FUNC();

   if ( !g_samplers[sidx].get() ) return ERROR_UNINIT;

   const ivec3 sz = g_samplers[sidx]->sampleSizes();
   float* d = dims;
   for ( size_t i = 0; i < Sampler::nBuffers; ++i )
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
   g_samplers[sidx].reset( new Sampler( datasetPath, dataPath, sz, seed ) );

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
