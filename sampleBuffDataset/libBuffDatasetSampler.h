/*! *****************************************************************************
 *   \file libBuffDatasetSampler.h
 *   \author moennen
 *   \brief
 *   \date 2018-04-06
 *   *****************************************************************************/

#ifndef _LIB_BUFF_DATASET_SAMPLER_H
#define _LIB_BUFF_DATASET_SAMPLER_H

enum
{
   SUCCESS = 0,
   ERROR_GENERIC,
   ERROR_BAD_ARGS,
   ERROR_BAD_DB,
   ERROR_UNINIT
};

extern "C" int getNbBuffers( const int sidx );

extern "C" int getBuffersDim( const int sidx, float* dims );

extern "C" int initBuffersDataSampler(
    const int sidx,
    const char* datasetPath,
    const char* dataPath,
    const int nParams,
    const float* params,
    const int seed );

extern "C" int getBuffersDataSample( const int sidx, float* buff );

#endif
