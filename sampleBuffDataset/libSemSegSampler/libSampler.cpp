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

#include "sampleBuffDataset/libSemSegSampler/datasetObjectsMapping.cpp"

// apply a map to transform the input labels to the output labels
// label mapping corresponds to the id of the maps contained in datasetObjectsMapping.h
// -1 means identity
void applyObjectMapping(const Mat& inLabels, Mat& outLabels, const int labelMapping)
{
  if (labelMapping < 0) 
  {
    inLabels.copyTo(outLabels); 
    return;
  }

  const vector<unsigned char>& map = getObjectsMapping( labelMapping );

  #pragma omp parallel for
   for ( unsigned y = 0; y < outLabels.rows; y++ )
   {
      const float* iLb = inLabels.ptr<float>( y );
      float* oLb = outLabels.ptr<float>( y );
      for ( unsigned x = 0; x < outLabels.cols; x++ )
      {
         const unsigned char lb = static_cast<unsigned char>(iLb[x]);
         oLb[x] = static_cast<float>(lb >= map.size() ? 0 : map[lb] );
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
   const int _objectMapping;

   enum
   {
      nBuffers = 2
   };
   ImgNFileLst _data;

   inline static unsigned getBufferDepth( const unsigned buffId ) { return buffId == 0 ? 3 : 1; }

   Sampler(
       const char* dataSetPath,
       const char* dataPath,
       const ivec3 sampleSz,
       const bool toLinear,
       const bool doRescale,
       const int objectMapping,
       const int seed )
       : _rng( seed ),
         _tsGen( 0.0, 1.0 ),
         _sampleSz( sampleSz ),
         _toLinear( toLinear ),
         _doRescale( doRescale ),
         _objectMapping( objectMapping ),
         _data( nBuffers )
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
         const size_t si = _dataGen( _rng );

         Mat currImg = cv_utils::imread32FC3( _data.filePath( si, 0 ), _toLinear, true /*toRGB*/ );
         Mat currLabels = cv_utils::imread32FC1( _data.filePath( si, 1 ), 1.0 );

         ivec2 imgSz( currImg.cols, currImg.rows );

         // ignore too small samples
         if ( ( imgSz.x < _sampleSz.y ) || ( imgSz.y < _sampleSz.z ) )
         {
            --s;
            continue;
         }

         // bad dataset : the samples have to be of the same size
         if ( ( currLabels.cols != imgSz.x ) || ( currLabels.rows != imgSz.y ) ) return false;

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
               // discreete scape
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
         Mat imgSple( _sampleSz.z, _sampleSz.y, CV_32FC3, currBuffImg );
         cv_utils::adjustContrastBrightness<vec3>(
             currImg, ( 1.0f + 0.11f * _rnGen( _rng ) ), 0.11f * _rnGen( _rng ) );
         GaussianBlur( currImg, imgSple, Size( 5, 5 ), 0.31f * abs( _rnGen( _rng ) ) );
         Mat labelsSple( _sampleSz.z, _sampleSz.y, CV_32FC1, currBuffLabels );
         // copy labels and apply a mapping if needed
         applyObjectMapping( currLabels, labelsSple, _objectMapping );

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
   const bool doRescale( nParams > 4 ? params[4] > 0.0 : true );
   const bool objectMapping( nParams > 5 ? static_cast<int>( params[5] ) : -1 );
   g_samplers[sidx].reset(
       new Sampler( datasetPath, dataPath, sz, toLinear, doRescale, objectMapping, seed ) );

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
