/*! *****************************************************************************
 *   \file sampleEnvMapShDataset_lib.cpp
 *   \author moennen
 *   \brief
 *   \date 2017-12-20
 *   *****************************************************************************/

#include "sampleEnvMapShDataset/sampleEnvMapShDataset_lib.h"
#include "utils/cv_utils.h"
#include "sampleEnvMapShDataset/envMapShDataSampler.h"

#include <glm/glm.hpp>
#include <leveldb/db.h>

#include <ctime>
#include <iostream>

using namespace std;
using namespace cv;

static std::unique_ptr<EnvMapShDataSampler> g_shSampler;

extern "C" int initEnvMapShDataSampler(const char* datasetName, const int shOrder)
{
   g_shSampler.reset();	
   {
      const string dbName(datasetName);
      leveldb::DB* db;
      leveldb::Options dbOpts;
      leveldb::Status dbStatus = leveldb::DB::Open( dbOpts, dbName, &db );
      if ( !dbStatus.ok() )
      {
         cerr << dbStatus.ToString() << endl;
         return SHS_ERROR_BAD_DB;
      }
      g_shSampler.reset( new EnvMapShDataSampler( shOrder, db, time( 0 ) ) );
   }

   return SHS_SUCCESS;
}

extern "C" int getEnvMapShNbCamParams() { return (g_shSampler.get()?g_shSampler->nbCameraParams():0); }
extern "C" int getEnvMapShNbCoeffs() { return (g_shSampler.get()?g_shSampler->nbShCoeffs():0); }

extern "C" int getEnvMapShDataSample(
   const int nbSamples,  
   float* shCoeffs, 
   float* camParams, 
   const int w, 
   const int h, 
   float* generatedViews)
{
   if (!g_shSampler.get()) return SHS_ERROR_UNINIT;

   // sample
   glm::uvec3 sz( w, h, nbSamples ); 
   if (!g_shSampler->sample( generatedViews, sz, shCoeffs, camParams ))
   {
      return SHS_ERROR_GENERIC;
   }

   return SHS_SUCCESS;
}
