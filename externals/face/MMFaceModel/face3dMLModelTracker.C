//*****************************************************************************/
//
// Filename face3dMLModelTracker.C
//
// Copyright (c) 2016 Autodesk, Inc.
// All rights reserved.
// 
// This computer source code and related instructions and comments are the 
// unpublished confidential and proprietary information of Autodesk, Inc.
// and are protected under applicable copyright and trade secret law.
// They may not be disclosed to, copied or used by any third party without 
// the prior written consent of Autodesk, Inc.
//*****************************************************************************/

#include <protoFace3d/face3dMLModelTracker.h>

#include <libFace3d/3dFaceModel01/MM_Restricted/FileLoader.h>
#include <libFace3d/3dFaceModel01/MM_Restricted/FileWriter.h>
#include <libFace3d/3dFaceModel01/MM_Restricted/DataContainer.h>

const size_t Face3dMLModelTracker::_landmarksIdx[13] = {
      3973, // left eye left corner
      2319, // left eye right corner
      4889, // right eye left corner 
      4609, // right eye right corner
      4302, // upper nose tip
      1344, // left nostril
      2636, // right nostril
      4775, // lower nose tip
      2291, // left mouth corner
      2677, // right mouth corner
      2296, // upper center mouth
      3548, // lower center mouth
      3893  // chin
};

Face3dMLModelTracker::Face3dMLModelTracker(
   const char *faceMeanModelFilename, 
   const char *faceMLModelFilename
) 
{
   // load trained statistical model.
   if (!_multilinearModelHandler.importMultilinearModel(faceMLModelFilename))
   {
      std::cerr << "MultiLinear Face Model not found " << faceMLModelFilename;
   }

   // load mean face model
   FileLoader loader;
   DataContainer meanFaceMesh;
   if (!loader.loadFile(faceMeanModelFilename, meanFaceMesh) || 
        meanFaceMesh.getNumFaces() <= 0 || 
        meanFaceMesh.getNumVertices()<=0)
   {
      std::cerr << "Mean face model not found " << faceMeanModelFilename;
   }
   else if (meanFaceMesh.getVertexIndexList()[0]->getDim() != 3)
   {
      std::cerr << "Faces mesh is not triangulated " << faceMeanModelFilename;
   }
   else 
   {
      _modelFaces.resize(meanFaceMesh.getNumFaces()*3);
      for(int i=0;i<meanFaceMesh.getNumFaces();++i)
      {
          _modelFaces[i*3] = (*(meanFaceMesh.getVertexIndexList()[i]))[0];
          _modelFaces[i*3+1] = (*(meanFaceMesh.getVertexIndexList()[i]))[1];
          _modelFaces[i*3+2] = (*(meanFaceMesh.getVertexIndexList()[i]))[2];
      }
   }
}

Face3dMLModelTracker::~Face3dMLModelTracker(){};

bool Face3dMLModelTracker::getTransformFromLandmarks(
   const TFaceLandmarks&,
   TFaceTransform &
)
{
   return true;
}

bool Face3dMLModelTracker::getMeanNeutralParams(
   TFaceParams &fp) const
{
   const std::vector<double> &p = 
      _multilinearModelHandler.getNeutralMeanWeights();
   fp = TFaceParams(p.size());
   std::memcpy(fp.data(),&p[0],p.size()*sizeof(double));
   return true;
}

bool Face3dMLModelTracker::getNeutralParamsFromLandmarks(
   const TFaceLandmarks&,
   TFaceParams &)
{
   return true;
}

bool Face3dMLModelTracker::updateFaceTexture( 
   const TImg &img,
   const TFaceTransform &,
   const TFaceParams &,
   float alpha,
   TImg &faceTexture )
{
   return true;
}
                        
bool Face3dMLModelTracker::getVerticesFromParams(
   const TFaceParams &params, 
   std::vector<float> &vertices)
{
   _multilinearModelHandler.reconstructForWeights(
      static_cast<const double *>(params.data()), vertices);
   return true;
}