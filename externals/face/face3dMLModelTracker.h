//*****************************************************************************/
//
// Filename face3dMLModelTracker.h
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
#ifndef _PROTOFACE3D_FACE3DMLMODELTRACKER_H
#define _PROTOFACE3D_FACE3DMLMODELTRACKER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <MM_Restricted/MultilinearModelHandler.h>
#include <GL/gl.h>
#include <vector>

#include <glm/glm.hpp>

class Face3dMLModelTracker
{
  public:
   typedef cv::Mat TImg;
   typedef Eigen::Matrix<double, 4, 4> TFaceTransform;
   typedef Eigen::VectorXd TFaceParams;
   typedef Eigen::Matrix<float, 13, 3> TFaceLandmarks;

  public:
   Face3dMLModelTracker( const char* faceMeanModelFilename, const char* faceMLModelFilename );
   virtual ~Face3dMLModelTracker();

   bool initialized() const { return _modelFaces.size(); }

   bool getTransformFromLandmarks( const TFaceLandmarks&, TFaceTransform& );
   bool getMeanNeutralParams( TFaceParams& ) const;
   bool getNeutralParamsFromLandmarks( const TFaceLandmarks&, TFaceParams& );

   bool updateFaceTexture(
       const TImg& img,
       const TFaceTransform&,
       const TFaceParams&,
       float alpha,
       TImg& faceTexture );

   bool getVerticesFromParams( const TFaceParams&, std::vector<glm::vec3>& );
   bool getVerticesFromNormParams( const TFaceParams&, std::vector<glm::vec3>& );
   
   const std::vector<glm::uvec3>& getFaces() const { return _modelFaces; }
   const std::vector<float>& getUVs() const { return _modelUVs; }

   inline size_t getShapeNCoeffs() const  {return _multilinearModelHandler.getTruncDim(1);}
   inline size_t getExprNCoeffs() const  {return _multilinearModelHandler.getTruncDim(2);}

   static std::array<size_t, 13> getLandmarksIdx() { return _landmarksIdx; }

  protected:
   MultilinearModelHandler _multilinearModelHandler;
   std::vector<glm::uvec3> _modelFaces;
   std::vector<float> _modelUVs;
   static const std::array<size_t, 13> _landmarksIdx;
};

#endif  // _PROTOFACE3D_FACE3DMLMODELTRACKER_H
