/*! *****************************************************************************
 *   \file shFromEnvMap_main.cpp
 *   \author moennen
 *   \brief
 *   \date 2017-12-07
 *   *****************************************************************************/

#include "utils/imgFileLst.h"
#include "utils/perf.h"

#include "sh/spherical_harmonics.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

// opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <omp.h>

#include <leveldb/db.h>

#include <string>
#include <boost/filesystem/convenience.hpp>

using namespace std;
using namespace cv;
using namespace glm;

const string keys =
    "{help h usage ? |            | print this message   }"
    "{@imgLst        |            | lst of image to extract the sh }"
    "{@db            |/tmp/out.db | sh coefficients file }";

namespace
{

template <int shOrder = 4>
bool computeSphericalHarmonics( Mat& img, vector<dvec3>& shCoeff )
{
   const int nbShCoeffs = sh::GetCoefficientCount( shOrder );

   // construct the sh degree/order indices
   static vector<ivec2> shLM;
   if ( shLM.size() != nbShCoeffs )
   {
      shLM.resize( nbShCoeffs );
      for ( int l = 0; l <= shOrder; ++l )
      {
         for ( int m = -l; m <= l; ++m )
         {
            shLM[sh::GetIndex( l, m )] = ivec2( l, m );
         }
      }
   }

   // integrate over the image
   const double pixel_area = ( 2.0 * M_PI / img.cols ) * ( M_PI / img.rows );
   shCoeff.resize( nbShCoeffs, dvec3( 0.0 ) );
   vector<vector<dvec3> > img_coeffs( img.rows, shCoeff );
#pragma omp parallel for
   for ( size_t y = 0; y < img.rows; y++ )
   {
      const float* row_data = img.ptr<float>( y );
      const double theta = sh::ImageYToTheta( y, img.rows );
      const double weight = pixel_area * sin( theta );
      vector<dvec3>& row_coeffs = img_coeffs[y];

      for ( size_t x = 0; x < img.cols; x++ )
      {
         const double phi = sh::ImageXToPhi( x, img.cols );
         // NB : opencv images use to be stored in BGR
         const dvec3 rgb( row_data[x * 3 + 2], row_data[x * 3 + 1], row_data[x * 3] );

         for ( int shi = 0; shi < nbShCoeffs; ++shi )
         {
            row_coeffs[shi] += weight * rgb * sh::EvalSH( shLM[shi].x, shLM[shi].y, phi, theta );
         }
      }
   }

   // set the output
   for ( size_t y = 0; y < img.rows; y++ )
   {
      const vector<dvec3>& row_coeffs = img_coeffs[y];
      for ( int shi = 0; shi < nbShCoeffs; ++shi )
      {
         shCoeff[shi] += row_coeffs[shi];
      }
   }

   return true;
}
}

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }
   string inputFilenameA = parser.get<string>( "@imgLst" );
   string outputFilename = parser.get<string>( "@db" );

   // create/open the database
   std::unique_ptr<leveldb::DB> dbPtr;
   {
      leveldb::DB* db;
      leveldb::Options dbOpts;
      dbOpts.create_if_missing = true;
      leveldb::Status dbStatus = leveldb::DB::Open( dbOpts, outputFilename, &db );
      if ( !dbStatus.ok() )
      {
         cerr << dbStatus.ToString() << endl;
         return -1;
      }
      dbPtr.reset( db );
   }

   // iterate over the list of files
   {
      Perf profiler;
      profiler.start();

      ImgFileLst lst( inputFilenameA.c_str() );
      leveldb::WriteOptions dbWriteOpts;
      for ( size_t lst_i = 0; lst_i < lst.size(); ++lst_i )
      {
         const string& imgPath = lst.get( lst_i );

         // read the image
         cv::Mat img = cv::imread( imgPath, IMREAD_UNCHANGED );
         if ( !img.data || (img.channels() < 3) || (img.channels() > 4) )
         {
            cerr << "Bad image " << imgPath << endl;
            continue;
         }
         if ( img.channels() == 4 ) cv::cvtColor( img, img, cv::COLOR_RGBA2RGB );
         if ( img.type() != CV_32F )
         {
            img.convertTo( img, CV_32F );
            img /= 255.0;
         }

         // compute the sh coefficients
         vector<dvec3> shCoeff;
         if ( !computeSphericalHarmonics( img, shCoeff ) )
         {
            cerr << "Cannot compute SH for " << imgPath << endl;
            continue;
         }

         // write the result to the database
         dbPtr->Put(
             dbWriteOpts,
             boost::filesystem::basename( imgPath ),
             leveldb::Slice(
                 reinterpret_cast<const char*>( value_ptr( shCoeff[0] ) ),
                 sizeof( dvec3 ) * shCoeff.size() ) );
      }

      profiler.stop();
      cout << "Sh computation time -> " << profiler.getMs() << endl;
   }

   // test database
   /*std::unique_ptr<leveldb::Iterator> dbIt( dbPtr->NewIterator( leveldb::ReadOptions() ) );
   for ( dbIt->SeekToFirst(); dbIt->Valid(); dbIt->Next() )
   {
      cout << dbIt->key().ToString() << " : " << dbIt->value().ToString() << endl;
   }*/

   return ( 0 );
}
