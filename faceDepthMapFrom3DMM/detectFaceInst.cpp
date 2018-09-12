/*!
 * *****************************************************************************
 *   \file faceDepthMapFrom3D_main.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-03-19
 *   *****************************************************************************/

#include "utils/imgFileLst.h"
#include "utils/cv_utils.h"

// face detector / models
#include "externals/face/faceDetector.h"

#include <glm/glm.hpp>

#include <memory>
#include <random>
#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;

void drawFaces( Mat& img, const vector<vec4>& faces )
{
   const Scalar colour(0.0, 1.0, 0.0, 0.85);
   const int thickness = 2.5;

   for ( const auto f : faces )
   {
      rectangle(img, Point(f.x,f.y), Point(f.z,f.w), colour, thickness, CV_AA);
   }
}

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@faceModel     |         | face detection model }"
    "{@imgFileLst    |         | image lst to detect face instances from  }"
    "{@imgRootDir    |         | images root dir   }"
    "{show           |         |    }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }
   const bool doShow = parser.get<bool>( "show" );

   // Face detector
   FaceDetector faceEngine;
   faceEngine.init( parser.get<string>( "@faceModel" ).c_str() );
   
   // Create the list of images
   ImgNFileLst imgLst(1,
       parser.get<string>( "@imgFileLst" ).c_str(), parser.get<string>( "@imgRootDir" ).c_str() );
   if ( imgLst.size() == 0 )
   {
      cerr << "Invalid dataset : " << parser.get<string>( "@imgFileLst" ) << endl;
      return -1;
   }

   unsigned startIdx = 0;
   const int nMaxRendersPerGroup = 10000;

   // Loop through the data
   for ( size_t i = 0; i < imgLst.size(); ++i )
   {
      // load the current image
      cv::Mat img = cv::imread( imgLst.filePath(i,0) );
      cv::cvtColor( img, img, cv::COLOR_BGR2RGB );

      vector<vec4> imgFaces;
      faceEngine.getFaces( img, imgFaces );
      
      cout << imgLst(i,0) << " " << imgFaces.size();



      for( const auto& faceInst : imgFaces  )
      {
        cout << " " << faceInst.x << " " << faceInst.y << " " << faceInst.z << " " << faceInst.w; 
      }

      cout << endl;

      if (doShow)
      {
         cv::cvtColor( img, img, cv::COLOR_BGR2RGB );
         drawFaces(img, imgFaces);
         imshow("faceInst", img);
         waitKey( 0 );
      }
   }

   return ( 0 );
}