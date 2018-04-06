/*! *****************************************************************************
 *   \file faceDetector.h
 *   \author 2018
 *   \brief
 *   \date 2018-04-06
 *   *****************************************************************************/
#ifndef _FACE_FACEDETECTOR_H
#define _FACE_FACEDETECTOR_H

#include <opencv2/imgproc.hpp>
#include <glm/glm.hpp>

#include <memory>
#include <vector>

class FaceDetector final
{
  public:
   FaceDetector();
   ~FaceDetector();

   bool init( const char* faceDetectorModel, const char* landmarksDetectorModel );

   void getFaces( cv::Mat& img, std::vector<glm::vec4> ) const;

   void getFacesLandmarks(
       cv::Mat& img,
       const size_t nbFaces,
       const glm::vec4* faces,
       std::vector<glm::vec2>* landmarks ) const;

  private:
   struct Detector;
   std::unique_ptr<Detector> _detectorPtr;
};

#endif  // _FACE_FACEDETECTOR_H
