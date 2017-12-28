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
SHS_SUCCESS,
SHS_ERROR_GENERIC,
SHS_ERROR_BAD_DB,
SHS_ERROR_UNINIT
};

extern "C" int initEnvMapShDataSampler(const char* datasetName, const int shOrder);

extern "C" int getEnvMapShNbCamParams();

extern "C" int getEnvMapShNbCoeffs();

extern "C" int getEnvMapShDataSample(const int nbSamples, float* shCoeffs, float* camParams, const int w, const int h, float* generatedView);

#endif
