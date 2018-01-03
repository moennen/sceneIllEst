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
SHS_SUCCESS=0,
SHS_ERROR_GENERIC,
SHS_ERROR_BAD_DB,
SHS_ERROR_UNINIT
};

extern "C" int initEnvMapShDataSampler(const int idx, const char* datasetName, const int shOrder);

extern "C" int getEnvMapShNbCamParams(const int idx);

extern "C" int getEnvMapShNbCoeffs(const int idx);

extern "C" int getEnvMapShDataSample(const int idx, const int nbSamples, float* shCoeffs, float* camParams, const int w, const int h, float* generatedView);

#endif
