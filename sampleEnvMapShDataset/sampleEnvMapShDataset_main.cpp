/*! *****************************************************************************
 *   \file sampleEnvMapShDataset_main.cpp
 *   \author moennen
 *   \brief
 *   \date 2017-12-20
 *   *****************************************************************************/

#include "utils/cv_utils.h"
#include "sampleEnvMapShDataset/envMapShDataSampler.h"

#include <glm/glm.hpp>
#include <leveldb/db.h>

#include <ctime>
#include <iostream>

using namespace std;
using namespace cv;

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@db            |         | leveldb database     }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }
   string outputFilename = parser.get<string>( "@db" );

   // open the database
   std::unique_ptr<EnvMapShDataSampler> shSampler;
   {
      leveldb::DB* db;
      leveldb::Options dbOpts;
      leveldb::Status dbStatus = leveldb::DB::Open( dbOpts, outputFilename, &db );
      if ( !dbStatus.ok() )
      {
         cerr << dbStatus.ToString() << endl;
         return -1;
      }
      shSampler.reset( new EnvMapShDataSampler( 8, db, std::time( 0 ) ) );
   }

   // sample
   glm::uvec3 sz( 128, 128, 10 );
   vector<float> imgData(sz.x*sz.y*sz.y*3);
   vector<float> camData(sz.z*shSampler->nbCameraParams());
   vector<float> shData(sz.z*shSampler->nbShCoeffs()*3);
   shSampler->sample(&imgData[0], sz, &shData[0], &camData[0]);

   return ( 0 );
}
