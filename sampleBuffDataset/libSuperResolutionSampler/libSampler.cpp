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
   boost::random::uniform_int_distribution<> _pathGen;
   boost::random::uniform_real_distribution<> _transGen;

   std::vector<std::string> _paths;
   const ivec3 _sampleSz;
   const bool _toLinear;
   const float _downsample;

   Sampler(
       const char* dataSetPath,
       const char* dataPath,
       const float downsampleFactor,
       const ivec3 sampleSz,
       const bool toLinear,
       const int seed )
       : _rng( seed ),
         _transGen( 0.0, 1.0 ),
         _sampleSz( sampleSz ),
         _toLinear( toLinear ),
         _downsample( downsampleFactor )
   {
      HOP_PROF_FUNC();

      const boost::filesystem::path rootPath( dataPath );
      std::ifstream ifs( dataSetPath );
      if ( ifs.is_open() )
      {
         _paths.reserve( 75000 );
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

      const unsigned buffSz = _sampleSz.z * _sampleSz.y * 3;
      float* currBuffHD = buff;
      float* currBuffLD = buff + buffSz * _sampleSz.x;

      for ( size_t s = 0; s < _sampleSz.x; ++s )
      {
         const std::string& iname = _paths[_pathGen( _rng )];

         Mat inputImg = cv_utils::imread32FC3( iname, _toLinear );
         ivec2 imgSz( inputImg.cols, inputImg.rows );

         // ignore too small samples
         if ( ( imgSz.x < _sampleSz.y ) || ( imgSz.y < _sampleSz.z ) )
         {
            --s;
            continue;
         }

         // random rescale
         const float ds =
             mix( 1.0f,
                  std::max( (float)_sampleSz.z / imgSz.y, (float)_sampleSz.y / imgSz.x ),
                  _transGen( _rng ) );
         resize( inputImg, inputImg, Size(), ds, ds, CV_INTER_AREA );
         imgSz = ivec2( inputImg.cols, inputImg.rows );

         // random translate
         const ivec2 trans(
             std::floor( _transGen( _rng ) * ( imgSz.x - _sampleSz.y ) ),
             std::floor( _transGen( _rng ) * ( imgSz.y - _sampleSz.z ) ) );

         // crop
         inputImg = inputImg( Rect( trans.x, trans.y, _sampleSz.y, _sampleSz.z ) );

         // random small blur to remove artifacts
         GaussianBlur( inputImg, inputImg, Size( 3, 3 ), 0.5 * _transGen( _rng ) );

         // bgr 2 rgb
         Mat sampleImg( _sampleSz.z, _sampleSz.y, CV_32FC3, currBuffHD );
         cvtColor( inputImg, sampleImg, COLOR_BGR2RGB );

         // downsample / upsample
         Mat sampleDsImg( _sampleSz.z, _sampleSz.y, CV_32FC3, currBuffLD );
         Mat tmpDsImg;
         resize( sampleImg, tmpDsImg, Size(), _downsample, _downsample, CV_INTER_AREA );
         resize( tmpDsImg, sampleDsImg, Size( _sampleSz.y, _sampleSz.z ), 0, 0, CV_INTER_LINEAR );

         currBuffHD += buffSz;
         currBuffLD += buffSz;
      }

      return true;
   }

   size_t nSamples() const { return _paths.size(); }
   ivec3 sampleSizes() const { return _sampleSz; }
};

array<unique_ptr<Sampler>, 33> g_samplers;
};

extern "C" int getNbBuffers( const int /*sidx*/ ) { return 2; }

extern "C" int getBuffersDim( const int sidx, float* dims )
{
   HOP_PROF_FUNC();

   if ( !g_samplers[sidx].get() ) return ERROR_UNINIT;

   ivec3 sz = g_samplers[sidx]->sampleSizes();
   dims[0] = sz.y;
   dims[1] = sz.z;
   dims[2] = 3;
   dims[3] = sz.y;
   dims[4] = sz.z;
   dims[5] = 3;

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
   const float downsampleFactor = params[3];
   const bool toLinear( nParams > 4 ? params[4] > 0.0 : false );
   g_samplers[sidx].reset(
       new Sampler( datasetPath, dataPath, downsampleFactor, sz, toLinear, seed ) );

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
