/*! *****************************************************************************
 *   \file libSampler.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-04-06
 *   *****************************************************************************/

#include "sampleBuffDataset/libBuffDatasetSampler.h"

#include "utils/cv_utils.h"
#include "utils/Hop.h"

#include <boost/filesystem.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

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

namespace
{
struct Sampler final
{
   const bool _rand;
   boost::random::mt19937 _rng;
   boost::random::uniform_int_distribution<> _pathGen;
   std::vector<std::string> _paths;
   const ivec3 _sampleSz;

   Sampler(
       const char* dataSetPath,
       const char* dataPath,
       const ivec3 sampleSz,
       const int seed,
       const bool doRand )
       : _rand( doRand ), _rng( seed ), _sampleSz( sampleSz )
   {
      HOP_PROF_FUNC();

      const boost::filesystem::path rootPath( dataPath );
      std::ifstream ifs( dataSetPath );
      if ( ifs.is_open() )
      {
         _paths.reserve( 25000 );
         std::string line;
         while ( ifs.good() )
         {
            getline( ifs, line );
            const boost::filesystem::path f( rootPath / boost::filesystem::path( line ) );
            if ( boost::filesystem::is_regular_file( f ) ) _paths.emplace_back( f.string() );
         }
         _paths.shrink_to_fit();
      }
      _pathGen = boost::random::uniform_int_distribution<>( 0, _paths.size() - 1 );
   }

   bool sample( float* buff )
   {
      HOP_PROF_FUNC();

      float* currBuff = buff;
      const unsigned buffOffset = _sampleSz.z * _sampleSz.y * 3;
      const uvec2 sampleImgSz(_sampleSz.y, _sampleSz.z);
      for ( size_t s = 0; s < _sampleSz.x; ++s )
      {
         Mat inputImg =
             cv_utils::imread32FC3( _paths[_rand ? _pathGen( _rng ) : s % _paths.size()] );
         Mat sampleImg( _sampleSz.z, _sampleSz.y, CV_32FC3, currBuff );
         cv_utils::fittResizeCrop( inputImg, sampleImgSz );
         inputImg.copyTo( sampleImg );
         cvtColor( sampleImg, sampleImg, COLOR_BGR2RGB );
         currBuff += buffOffset;
      }

      return true;
   }

   size_t nSamples() const { return _paths.size(); }
   ivec3 sampleSizes() const { return _sampleSz; }
};

array<unique_ptr<Sampler>, 33> g_samplers;
};

extern "C" int getNbBuffers( const int /*sidx*/ ) { return 1; }

extern "C" int getBuffersDim( const int sidx, float* dims )
{
   HOP_PROF_FUNC();

   if ( !g_samplers[sidx].get() ) return ERROR_UNINIT;

   ivec3 sz = g_samplers[sidx]->sampleSizes();
   dims[0] = sz.y;
   dims[1] = sz.z;
   dims[2] = 3;

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
   const bool doRand( nParams > 3 ? params[3] != 0.0 : false );
   g_samplers[sidx].reset( new Sampler( datasetPath, dataPath, sz, seed, doRand ) );

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
