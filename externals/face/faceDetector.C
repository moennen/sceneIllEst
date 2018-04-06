/*! *****************************************************************************
 *   \file faceDetector.C
 *   \author 2018
 *   \brief
 *   \date 2018-04-06
 *   *****************************************************************************/

#include <face/faceDetector.h>

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>

using namespace dlib;

class FaceDetector::Detector final
{
   // dlib face detector
   template <long num_filters, typename SUBNET>
   using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
   template <long num_filters, typename SUBNET>
   using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

   template <typename SUBNET>
   using downsampler =
       relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
   template <typename SUBNET>
   using rcon5 = relu<affine<con5<45, SUBNET>>>;

   using net_type = loss_mmod<
       con<1,
           9,
           9,
           1,
           1,
           rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

   net_type _net;

  public:
   Detector();
   ~Detector();

   bool init( const char* faceModel, const char* landmarksModel );

   void getFaces( cv::Mat& img, std::vector<glm::vec4> );

   void getFacesLandmarks(
       cv::Mat& img,
       const size_t nbFaces,
       const glm::vec4* face,
       std::vector<glm::vec2>* landmarks );
};

FaceDetector::Detector::Detector() {}
FaceDetector::Detector::~Detector() {}

bool FaceDetector::Detector::init( const char* faceModel, const char* landmarksModel )
{
   deserialize( faceModel ) >> _net;
   return true;
}

void FaceDetector::Detector::getFaces( cv::Mat& cvimg, std::vector<glm::vec4> faces )
{
   // convert to dlib
   array2d<rgb_pixel> img;
   assign_image( img, cv_image<rgb_pixel>( cvimg ) );

   // upsample the image to detect low res faces
   // while ( img.size() < 2048 * 2048 ) pyramid_up( img );

   // run the model
   auto dets = _net( img );

   // transform from dlib rectangle to vec4 roi
   faces.reserve( dets[0].size() );
   for ( size_t i = 0; i < dets[0].size(); ++i )
   {
      const auto faceRect = dets[0][i];
      faces.emplace_back( faceRect.rect.left(), faceRect.rect.bottom(), faceRect.rect.right(), faceRect.rect.top() );
   }
}

void FaceDetector::Detector::getFacesLandmarks(
    cv::Mat& /*img*/,
    const size_t /*nbFaces*/,
    const glm::vec4* /*faces*/,
    std::vector<glm::vec2>* /*landmarks*/ )
{
}

FaceDetector::FaceDetector() : _detectorPtr( new Detector() )
{
};

FaceDetector::~FaceDetector(){};

bool FaceDetector::init( const char* faceModel, const char* landmarksModel )
{
   return _detectorPtr->init( faceModel, landmarksModel );
}

void FaceDetector::getFaces( cv::Mat& img, std::vector<glm::vec4> faces ) const
{
   _detectorPtr->getFaces( img, faces );
}

void FaceDetector::getFacesLandmarks(
    cv::Mat& img,
    const size_t nbFaces,
    const glm::vec4* faces,
    std::vector<glm::vec2>* landmarks ) const
{
   _detectorPtr->getFacesLandmarks( img, nbFaces, faces, landmarks );
}
