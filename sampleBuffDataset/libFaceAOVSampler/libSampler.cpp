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

   static constexpr float maxDsScaleFactor = 3.5;

   const ivec3 _sampleSz;
   const bool _toLinear;
   const bool _doRescale;

   enum
   {
      nBuffers = 4
   };
   ImgNFileLst _data;

   inline static unsigned getBufferDepth( const unsigned buffId )
   {
      if ( buffId == 1 )
         return 2;
      else if ( buffId == 2 )
         return 1;
      else
         return 3;
   }

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
         _data( nBuffers - 1 )
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

      const unsigned depthBuffSz = _sampleSz.y * _sampleSz.z;
      const unsigned uvBuffSz = depthBuffSz * 2;
      const unsigned imgBuffSz = depthBuffSz * 3;

      float* currBuffImg = buff;
      float* currBuffUVs = buff + imgBuffSz * _sampleSz.x;
      float* currBuffDepth = buff + ( imgBuffSz + uvBuffSz ) * _sampleSz.x;
      float* currBuffNormals = buff + ( imgBuffSz + uvBuffSz + depthBuffSz ) * _sampleSz.x;

      for ( size_t s = 0; s < _sampleSz.x; ++s )
      {
         const size_t si = _dataGen( _rng );

         Mat currImg = cv_utils::imread32FC3( _data.filePath( si, 0 ), _toLinear, true /*toRGB*/ );
         Mat currUVDepth = cv_utils::imread32FC3( _data.filePath( si, 1 ), false, true );
         Mat currNormals = cv_utils::imread32FC3( _data.filePath( si, 2 ), false, true );

         ivec2 imgSz( currImg.cols, currImg.rows );

         // padd too small samples
         if ( ( imgSz.x < _sampleSz.y ) || ( imgSz.y < _sampleSz.z ) )
         {
            const ivec2 topright(
                max( ivec2( _sampleSz.y - imgSz.x, _sampleSz.z - imgSz.y ), ivec2( 0 ) ) );
            copyMakeBorder( currImg, currImg, topright.y, 0, 0, topright.x, BORDER_CONSTANT );
            copyMakeBorder( currUVDepth, currUVDepth, topright.y, 0, 0, topright.x, BORDER_CONSTANT );
            copyMakeBorder( currNormals, currNormals, topright.y, 0, 0, topright.x, BORDER_CONSTANT );
            imgSz = ivec2( currImg.cols, currImg.rows );
         }

         // bad dataset : the samples o be of the same size
         if ( ( currUVDepth.cols != imgSz.x ) || ( currUVDepth.rows != imgSz.y ) ) return false;
         if ( ( currNormals.cols != imgSz.x ) || ( currNormals.rows != imgSz.y ) ) return false;

         // random rescale
         if ( _doRescale )
         {
            const float minDs =
                std::max( (float)_sampleSz.z / imgSz.y, (float)_sampleSz.y / imgSz.x );
            const float ds =
                mix( std::min( 1.0f, maxDsScaleFactor * minDs ), minDs, _tsGen( _rng ) );
            resize( currImg, currImg, Size(), ds, ds, CV_INTER_AREA );
            resize( currUVDepth, currUVDepth, Size(), ds, ds, CV_INTER_AREA );
            resize( currNormals, currNormals, Size(), ds, ds, CV_INTER_AREA );
            imgSz = ivec2( currImg.cols, currImg.rows );
         }

         // random translate
         const ivec2 trans(
             std::floor( _tsGen( _rng ) * ( imgSz.x - _sampleSz.y ) ),
             std::floor( _tsGen( _rng ) * ( imgSz.y - _sampleSz.z ) ) );

         // crop
         currImg = currImg( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );
         currUVDepth = currUVDepth( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );
         currNormals = currNormals( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );

         // random small blur to remove artifacts + copy to destination
         Mat imgSple( _sampleSz.z, _sampleSz.y, CV_32FC3, currBuffImg );
         cv_utils::adjustContrastBrightness<vec3>(
             currImg, ( 1.0f + 0.11f * _rnGen( _rng ) ), 0.11f * _rnGen( _rng ) );
         GaussianBlur( currImg, imgSple, Size( 5, 5 ), 0.31 * abs( _rnGen( _rng ) ) );

         // split uvdepth and
         Mat uvd[3];
         split( currUVDepth, uvd );

         Mat uvsSple( _sampleSz.z, _sampleSz.y, CV_32FC2, currBuffUVs );
         Mat uvl;
         merge( &uvd[0], 2, uvl );
         uvl.copyTo( uvsSple );

         Mat depthSple( _sampleSz.z, _sampleSz.y, CV_32FC1, currBuffDepth );
         uvd[2].copyTo( depthSple );

         // normalize the normals
         Mat normalsSple( _sampleSz.z, _sampleSz.y, CV_32FC3, currBuffNormals );
         currNormals.copyTo( normalsSple );

         currBuffImg += imgBuffSz;
         currBuffDepth += depthBuffSz;
         currBuffUVs += uvBuffSz;
         currBuffNormals += imgBuffSz;
      }

      return true;
   }

   size_t nSamples() const { return _data.size(); }
   ivec3 sampleSizes() const { return _sampleSz; }

   // normalize normal by z to reduce the dimension from 3 to 2
   void processNormals( const Mat& in, Mat& out )
   {
#pragma omp parallel for
      for ( size_t y = 0; y < in.rows; y++ )
      {
         const vec3* in_data = in.ptr<vec3>( y );
         vec2* out_data = out.ptr<vec2>( y );
         for ( size_t x = 0; x < in.cols; x++ )
         {
            const vec3& norm = in_data[x];
            out_data[x] =
                ( norm.z > 0.000001 ? ( vec2( norm.x, norm.y ) / norm.z ) : vec2( 0.0, 0.0 ) );
         }
      }
   }
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
