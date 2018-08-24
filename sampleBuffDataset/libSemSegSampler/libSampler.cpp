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

#include <iostream>
#include <array>

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

   const ivec3 _sampleSz;
   const bool _toLinear;

   enum
   {
      nBuffers = 2
   };
   ImgNFileLst<nBuffers> _data;

   inline static unsigned getBufferDepth( const unsigned buffId ) { return buffId == 0 ? 3 : 1; }

   Sampler(
       const char* dataSetPath,
       const char* dataPath,
       const ivec3 sampleSz,
       const bool toLinear,
       const int seed )
       : _rng( seed ), _tsGen( 0.0, 1.0 ), _sampleSz( sampleSz ), _toLinear( toLinear )
   {
      HOP_PROF_FUNC();

      _data.open( dataSetPath, dataPath );

      if ( _data.size() )
      {
         cout << "Read dataset " << dataSetPath << " (" << _data.size() << ") " << endl;
      }
      _dataGen = uniform_int_distribution<>( 0, _data.size() - 1 );
   }

   bool sample( float* buff )
   {
      HOP_PROF_FUNC();

      const unsigned labelsBuffSz = _sampleSz.y * _sampleSz.z;
      const unsigned imgBuffSz = labelsBuffSz * 3;

      float* currBuffImg = buff;
      float* currBuffLabels = buff + imgBuffSz * _sampleSz.x;

      for ( size_t s = 0; s < _sampleSz.x; ++s )
      {
         const ImgNFileLst<nBuffers>::Data& data = _data[_dataGen( _rng )];

         Mat currImg = cv_utils::imread32FC3( data[0], _toLinear, true /*toRGB*/ );
         Mat currLabels = cv_utils::imread32FC1( data[1] );

         ivec2 imgSz( currImg.cols, currImg.rows );

         // ignore too small samples
         if ( ( imgSz.x < _sampleSz.y ) || ( imgSz.y < _sampleSz.z ) )
         {
            --s;
            continue;
         }

         // bad dataset : the samples have to be of the same size
         if ( ( currLabels.cols != imgSz.x ) || ( currLabels.rows != imgSz.y ) ) return false;

         // random translate
         const ivec2 trans(
             std::floor( _tsGen( _rng ) * ( imgSz.x - _sampleSz.y ) ),
             std::floor( _tsGen( _rng ) * ( imgSz.y - _sampleSz.z ) ) );

         // crop
         currImg = currImg( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );
         currLabels = currLabels( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );

         // random small blur to remove artifacts + copy to destination
         Mat imgSple( _sampleSz.z, _sampleSz.y, CV_32FC3, currBuffImg );
         GaussianBlur( currImg, imgSple, Size( 5, 5 ), 1.5 * _tsGen( _rng ) );
         Mat labelsSple( _sampleSz.z, _sampleSz.y, CV_32FC1, currBuffLabels );
         currLabels.copyTo( labelsSple );

         currBuffImg += imgBuffSz;
         currBuffLabels += labelsBuffSz;
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
   const bool toLinear( nParams > 3 ? params[3] > 0.0 : false );
   g_samplers[sidx].reset( new Sampler( datasetPath, dataPath, sz, toLinear, seed ) );

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
