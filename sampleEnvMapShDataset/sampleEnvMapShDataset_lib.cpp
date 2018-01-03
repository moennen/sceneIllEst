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
#include <map>

using namespace std;
using namespace cv;

static map<int,  unique_ptr<EnvMapShDataSampler> > g_shSampler;

extern "C" int initEnvMapShDataSampler(const int idx, const char* datasetName, const int shOrder)
{
   g_shSampler[idx].reset();	
   {
      const string dbName(datasetName);
      leveldb::DB* db;
      leveldb::Options dbOpts;
      leveldb::Status dbStatus = leveldb::DB::Open( dbOpts, dbName, &db );
      if ( !dbStatus.ok() )
      {
         g_shSampler.erase(idx);	
         cerr << dbStatus.ToString() << endl;
         return SHS_ERROR_BAD_DB;
      }
      g_shSampler[idx].reset(
         new EnvMapShDataSampler( shOrder, db, time( 0 ) ) );
   }

   return SHS_SUCCESS;
}

extern "C" int getEnvMapShNbCamParams(const int idx) { 
   return (g_shSampler[idx].get()?g_shSampler[idx]->nbCameraParams():0); }
extern "C" int getEnvMapShNbCoeffs(const int idx) { 
   return (g_shSampler[idx].get()?g_shSampler[idx]->nbShCoeffs():0); }

extern "C" int getEnvMapShDataSample(
   const int idx,
   const int nbSamples,  
   float* shCoeffs, 
   float* camParams, 
   const int w, 
   const int h, 
   float* generatedViews)
{
   if (!g_shSampler[idx].get()) return SHS_ERROR_UNINIT;

   // sample
   glm::uvec3 sz( w, h, nbSamples ); 
   if (!g_shSampler[idx]->sample( generatedViews, sz, shCoeffs, camParams ))
   {
      return SHS_ERROR_GENERIC;
   }

   return SHS_SUCCESS;
}
