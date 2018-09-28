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

   const ivec3 _sampleSz;
   const bool _toLinear;
   const bool _doRescale;

   const unsigned _fullBuffSz;

   future<bool> _asyncSample;
   vector<float> _asyncBuff;

   enum
   {
      nBuffers = 5,
      nOutPlanes = 13  // RGB : 3 / UVD : 3 / Normals : 3 / ID : 2 / Pos : 2
   };
   ImgNFileLst _data;

   inline static unsigned getBufferDepth( const unsigned buffId ) { return buffId < 3 ? 3 : 2; }

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
         _data( nBuffers - 1 )
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
      const size_t szBuffC2 = _sampleSz.y * _sampleSz.z * 2;

      const size_t szBuffOffUVD = _sampleSz.x * szBuffC3;
      const size_t szBuffOffNorm = szBuffOffUVD + _sampleSz.x * szBuffC3;
      const size_t szBuffOffID = szBuffOffNorm + _sampleSz.x * szBuffC3;
      const size_t szBuffOffPos = szBuffOffID + _sampleSz.x * szBuffC2;

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

            Mat currImg =
                cv_utils::imread32FC3( _data.filePath( si, 0 ), _toLinear, true /*toRGB*/ );
            Mat currUVDepth = cv_utils::imread32FC3( _data.filePath( si, 1 ), false, true );
            Mat currNormals = cv_utils::imread32FC3( _data.filePath( si, 2 ), false, true );
            Mat currIDMatte = cv_utils::imread32FC3( _data.filePath( si, 3 ), false, true );

            // ignore failed samples
            if ( currImg.empty() || currUVDepth.empty() || currNormals.empty() ||
                 currIDMatte.empty() )
               continue;

            ivec2 imgSz( currImg.cols, currImg.rows );

            // padd too small samples
            if ( ( imgSz.x < _sampleSz.y ) || ( imgSz.y < _sampleSz.z ) )
            {
               const ivec2 topright(
                   max( ivec2( _sampleSz.y - imgSz.x, _sampleSz.z - imgSz.y ), ivec2( 0 ) ) );
               copyMakeBorder( currImg, currImg, topright.y, 0, 0, topright.x, BORDER_CONSTANT );
               copyMakeBorder(
                   currUVDepth, currUVDepth, topright.y, 0, 0, topright.x, BORDER_CONSTANT );
               copyMakeBorder(
                   currNormals, currNormals, topright.y, 0, 0, topright.x, BORDER_CONSTANT );
               copyMakeBorder(
                   currIDMatte, currIDMatte, topright.y, 0, 0, topright.x, BORDER_CONSTANT );
               imgSz = ivec2( currImg.cols, currImg.rows );
            }

            // random rescale
            float fRescaleFactor = 1.0f;
            if ( _doRescale )
            {
               const float minDs =
                   std::max( (float)_sampleSz.z / imgSz.y, (float)_sampleSz.y / imgSz.x );
               fRescaleFactor = mix( std::min( 1.0f, minDs ), minDs, _tsGen( _rng ) );
               resize( currImg, currImg, Size(), fRescaleFactor, fRescaleFactor, CV_INTER_AREA );
               resize(
                   currUVDepth,
                   currUVDepth,
                   Size(),
                   fRescaleFactor,
                   fRescaleFactor,
                   CV_INTER_AREA );
               resize(
                   currNormals,
                   currNormals,
                   Size(),
                   fRescaleFactor,
                   fRescaleFactor,
                   CV_INTER_AREA );
               resize(
                   currIDMatte,
                   currIDMatte,
                   Size(),
                   fRescaleFactor,
                   fRescaleFactor,
                   CV_INTER_AREA );
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
            currIDMatte = currIDMatte( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );

            // random small blur to remove artifacts + copy to destination
            Mat imgSple( _sampleSz.z, _sampleSz.y, CV_32FC3, buff + s * szBuffC3 );
            cv_utils::adjustContrastBrightness<vec3>(
                currImg, ( 1.0f + 0.11f * _rnGen( _rng ) ), 0.11f * _rnGen( _rng ) );
            GaussianBlur( currImg, imgSple, Size( 5, 5 ), 0.31 * abs( _rnGen( _rng ) ) );

            // uvdepth
            Mat uvdSple( _sampleSz.z, _sampleSz.y, CV_32FC3, buff + szBuffOffUVD + s * szBuffC3 );
            currUVDepth.copyTo( uvdSple );

            // normalize the normals
            Mat normalSple(
                _sampleSz.z, _sampleSz.y, CV_32FC3, buff + szBuffOffNorm + s * szBuffC3 );
            currNormals.copyTo( normalSple );

            // idmatte :
            Mat idSple( _sampleSz.z, _sampleSz.y, CV_32FC2, buff + szBuffOffID + s * szBuffC2 );
            Mat posSple( _sampleSz.z, _sampleSz.y, CV_32FC2, buff + szBuffOffPos + s * szBuffC2 );
            const vec2 v2PosScale = 
                max( vec2( 0.001f, 0.001f ), vec2( _tsGen( _rng ), _tsGen( _rng ) ) );
            const vec2 v2PosOff = 
                ( vec2( _tsGen( _rng ), _tsGen( _rng ) ) * 2.0f - 1.0f ) * v2PosScale;
            const vec2 v2IdScale = vec2( fRescaleFactor ) * v2PosScale;
            const vec2 v2IdOff =
                vec2( -2.0f * trans.x / imgSz.x + 1.0f, -2.0f * trans.y / imgSz.y + 1.0f ) *
                    v2PosScale +
                v2PosOff;

            processIdPos( currIDMatte, idSple, posSple, v2IdScale, v2IdOff, v2PosScale, v2PosOff );

            sampled[s] = 1;
         }
      } while ( accumulate( sampled.begin(), sampled.end(), 0 ) != _sampleSz.x );

      return true;
   }

   // set the id map
   void processIdPos(
       const Mat& matIdMatteIn,
       Mat& matIdOut,
       Mat& matPosOut,
       const vec2& v2IdScale,
       const vec2& v2IdOff,
       const vec2& v2PosScale,
       const vec2& v2PosOff )
   {
      const vec2 v2ScalePos = vec2( 2.0f / matIdMatteIn.cols, 2.0f / matIdMatteIn.rows );
#pragma omp parallel for
      for ( size_t y = 0; y < matIdMatteIn.rows; y++ )
      {
         const vec3* v3pIdMatteIn = matIdMatteIn.ptr<vec3>( y );
         vec2* v2pIdOut = matIdOut.ptr<vec2>( y );
         vec2* v2pPosOut = matPosOut.ptr<vec2>( y );
         for ( size_t x = 0; x < matIdMatteIn.cols; x++ )
         {
            const vec3& v3IdMatteIn = v3pIdMatteIn[x];
            v2pIdOut[x] = v3IdMatteIn.z > 0.5 ? vec2( v3IdMatteIn.x, v3IdMatteIn.y ) * v2IdScale + v2IdOff : vec2(-1.0);
            v2pPosOut[x] = ( vec2( x, y ) * v2ScalePos - 1.0f ) * v2PosScale + v2PosOff;
         }
      }
   }

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
   // WARNING : rescaling may cause unwanted artifact in normal  uvd maps
   const bool doRescale( nParams > 4 ? params[4] > 0.0 : false );
   const bool doAsync( nParams > 5 ? params[5] > 0.0 : true );
   g_samplers[sidx].reset(
       new Sampler( datasetPath, dataPath, sz, toLinear, doRescale, doAsync, seed ) );

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
