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
    "{@db            |         | leveldb database     }"
    "{@imgDir        |         | image root dir       }"
    "{@imgName       |         | test image name      }"
    "{order          |8        | sh order             }"
    "{cs             |0=sRGB 1=linear sRGB| colorSpace}";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }
   const string& dbFilename = parser.get<string>( "@db" );
   const string& imgRootDir = parser.get<string>( "@imgDir" );
   const string& imgFilename = parser.get<string>( "@imgName" );
   const int shOrder = parser.get<int>( "order" );
   const int cs = parser.get<int>( "cs" );

   // open the database
   std::unique_ptr<EnvMapShDataSampler> shSampler;
   {
      leveldb::DB* db;
      leveldb::Options dbOpts;
      leveldb::Status dbStatus = leveldb::DB::Open( dbOpts, dbFilename, &db );
      if ( !dbStatus.ok() )
      {
         cerr << dbStatus.ToString() << endl;
         return -1;
      }
      shSampler.reset( new EnvMapShDataSampler( shOrder, db, imgRootDir, std::time( 0 ), cs ) );
   }

   // sample
   glm::uvec3 sz( 256, 128, 1 );
   vector<float> imgData( sz.x * sz.y * sz.y * 3, 1.0 );
   Mat img( sz.y, sz.x, CV_32FC3, &imgData[0], Mat::AUTO_STEP );
   vector<float> camData( sz.z * shSampler->nbCameraParams() );
   vector<float> shData( sz.z * shSampler->nbShCoeffs() * 3 );
   bool success = shSampler->sample( &imgData[0], sz, &shData[0], &camData[0] );
   if ( !success )
   {
      cerr << "EnvMapShDataSampler::sample" << endl;
      return ( -1 );
   }
   imshow( "Sample", img );

   // generate envMap
   success = EnvMapShDataSampler::nbShCoeffs( shOrder ) == shSampler->nbShCoeffs();
   if ( !success )
   {
      cerr << "EnvMapShDataSampler::nbShCoeffs" << endl;
      return ( -1 );
   }
   success =
       EnvMapShDataSampler::loadSampleImg( imgFilename.c_str(), &imgData[0], sz.x, sz.y, cs );
   if ( !success )
   {
      cerr << "EnvMapShDataSampler::loadSampleImg" << endl;
      return ( -1 );
   }
   success = EnvMapShDataSampler::generateEnvMapFromShCoeffs(
       shOrder, &shData[0], &imgData[0], sz.x, sz.y );
   if ( !success )
   {
      cerr << "EnvMapShDataSampler::generateEnvMapFromShCoeffs" << endl;
      return ( -1 );
   }
   imshow( "EnvMap", img );
   waitKey( 0 );

   return ( 0 );
}
