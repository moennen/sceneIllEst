/*! *****************************************************************************
 *   \file sampleEnvMapShDataset_lib.h
 *   \author moennen
 *   \brief
 *   \date 2017-12-20
 *   *****************************************************************************/

#ifndef _SAMPLEENVMAPSHDATASET_LIB_H
#define _SAMPLEENVMAPSHDATASET_LIB_H

enum
{
   SHS_SUCCESS = 0,
   SHS_ERROR_GENERIC,
   SHS_ERROR_BAD_DB,
   SHS_ERROR_UNINIT
};

extern "C" int
getImgFromFile( const char* fileName, float* img, const int w, const int h, const bool linearCS );

extern "C" int getNbShCoeffs( const int shOrder );

extern "C" int getEnvMapFromCoeffs(
    const int shOrder,
    const float* shCoeffs,
    float* envMap,
    const int w,
    const int h );

extern "C" int initEnvMapShDataSampler(
    const int idx,
    const char* datasetName,
    const char* imgRootDir,
    const int shOrder,
    const int seed,
    const bool linearCS );

extern "C" int getEnvMapShNbCamParams( const int idx );

extern "C" int getEnvMapShNbCoeffs( const int idx );

extern "C" int getEnvMapShDataSample(
    const int idx,
    const int nbSamples,
    float* shCoeffs,
    float* camParams,
    const int w,
    const int h,
    float* generatedView );

#endif
