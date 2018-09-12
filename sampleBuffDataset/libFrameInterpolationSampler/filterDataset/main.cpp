/*! *****************************************************************************
 *   \file filterDataset_main.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-09-07
 *   *****************************************************************************/

#include "utils/imgFileLst.h"
#include "utils/cv_utils.h"

#include <Eigen/Dense>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <boost/filesystem.hpp>

#include <limits>
#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;
using namespace boost;
using namespace Eigen;

namespace
{
//------------------------------------------------------------------------------
//
void resizeToMax( Mat& img, const uvec2 sampleSz )
{
   const uvec2 imgSz( img.cols, img.rows );
   // random rescale
   const float ds = std::max( (float)sampleSz.y / imgSz.y, (float)sampleSz.x / imgSz.x );
   if ( ( ds > 0.0 ) && ( ds < 1.0 ) ) resize( img, img, Size(), ds, ds, CV_INTER_AREA );
}
};

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@imgFileLst   |         | images list   }"
    "{@imgRootDir   |         | images root dir   }"
    "{@imgOutDir    |         | images output root dir   }"
    "{maxWidth      |640      | maximum width of the output image }"
    "{maxHeight     |480      | maximum heigth of the output image }"
    "{minError      |0.005    | minimum error }"
    "{maxError      |0.015    | maximum error }"
    "{show          |         |   }"
    "{write         |         | write images   }"
    "{stats         |         | compute error stats   }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }
   const filesystem::path outRootPath( parser.get<string>( "@imgOutDir" ) );

   const bool doShow = parser.get<bool>( "show" );
   const bool doWrite = parser.get<bool>( "write" );
   const bool doStats = parser.get<bool>( "stats" );
   const double minErr = parser.get<double>( "minError" );
   const double maxErr = parser.get<double>( "maxError" );
   const uvec2 maxSize( parser.get<int>( "maxWidth" ), parser.get<int>( "maxHeight" ) );
   double statMin = numeric_limits<double>::max();
   double statMax = numeric_limits<double>::min();
   double statMean = 0.0;
   size_t nValid = 0;

   // Create the list of image triplets + alpha
   ImgNFileLst imgLst(
       4,
       parser.get<string>( "@imgFileLst" ).c_str(),
       parser.get<string>( "@imgRootDir" ).c_str() );
   if ( imgLst.size() == 0 )
   {
      cerr << "Invalid dataset : " << parser.get<string>( "@imgFileLst" ) << endl;
      return -1;
   }

   for ( size_t i = 0; i < imgLst.size(); ++i )
   {
      Mat prevImg = cv_utils::imread32FC3( imgLst.filePath( i, 0 ) );
      resizeToMax(prevImg, maxSize);
      Mat nextImg = cv_utils::imread32FC3( imgLst.filePath( i, 2 ) );
      resizeToMax(nextImg, maxSize);

      const double merr = sqrt( norm( prevImg, nextImg ) / ( prevImg.cols * prevImg.rows * 3 ) );
      if ( ( merr > maxErr ) || ( merr < minErr ) )
      {
         continue;
      }

      const auto& data = imgLst[i];

      if ( doStats )
      {
         nValid++;
         statMin = std::min( statMin, merr );
         statMax = std::max( statMax, merr );
         statMean += merr;

         cout << merr << " -> " << statMean / nValid << "  [ " << statMin << " , " << statMax
              << "] " << (double)nValid / ( i + 1 ) << endl;
      }
      else
      {
         cout << data[0] << " " << data[1] << " " << data[2] << " " << data[3] << endl;
      }

      if ( doShow )
      {
         imshow( "InterpBlend", ( prevImg + nextImg ) * 0.5f );
         waitKey( 0 );
      }

      if (doWrite)
      {
        Mat currImg = cv_utils::imread32FC3( imgLst.filePath( i, 2 ) );
        resizeToMax(currImg, maxSize);

        imwrite( filesystem::path( outRootPath / data[0] ).string().c_str(), prevImg * 255.0 );
        imwrite( filesystem::path( outRootPath / data[1] ).string().c_str(), currImg * 255.0 );
        imwrite( filesystem::path( outRootPath / data[2] ).string().c_str(), nextImg * 255.0 );
      }
   }

   return ( 0 );
}
