/*! *****************************************************************************
 *   \file shFromEnvMap_main.cpp
 *   \author moennen
 *   \brief
 *   \date 2017-12-07
 *   *****************************************************************************/

#include "utils/imgFileLst.h"
#include "utils/perf.h"
#include "utils/cv_utils.h"

#include "sh/spherical_harmonics.h"

#include <Eigen/Dense>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

#include <iostream>
#include <omp.h>

#include <leveldb/db.h>

#include <string>
#include <boost/filesystem/convenience.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace glm;

const string keys =
    "{help h usage ? |            | print this message   }"
    "{@db            |/tmp/shCoeffs.db | sh coefficients file }"
    "{mode           |0           | 0=compute coeffs, 1=estimate coeffs statistics  }"
    "{shOrder        |8           | sh order }"
    "{imgLst         |            | lst of image to extract the sh }";

namespace
{
bool testEnvMapDir( Mat& img )
{
#pragma omp parallel for
   for ( size_t y = 0; y < img.rows; y++ )
   {
      float* row_data = img.ptr<float>( y );
      const double theta = ( 0.5 - ( y + 0.5 ) / img.rows ) * M_PI;
      const double stheta = sin( theta );
      const double ctheta = cos( theta );

      for ( size_t x = 0; x < img.cols; x++ )
      {
         const double phi = ( 2.0 * ( x + 0.5 ) / img.cols - 1.0 ) * M_PI;
         Vector3d dir;
         dir << ctheta * sin( phi ), stheta, ctheta * cos( phi );
         // NB : opencv images use to be stored in BGR
         row_data[x * 3 + 2] = dir[0];
         row_data[x * 3 + 1] = dir[1];
         row_data[x * 3 + 0] = dir[2];
      }
   }
   return true;
}

bool computeSphericalHarmonics( Mat& img, vector<dvec3>& shCoeff, const int shOrder )
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
      const double theta = ( 0.5 - ( y + 0.5 ) / img.rows ) * M_PI;
      const double stheta = sin( theta );
      const double ctheta = cos( theta );
      const double weight = pixel_area * ctheta;
      vector<dvec3>& row_coeffs = img_coeffs[y];
      for ( size_t x = 0; x < img.cols; x++ )
      {
         const double phi = ( 2.0 * ( x + 0.5 ) / img.cols - 1.0 ) * M_PI;
         Vector3d dir;
         dir << ctheta * sin( phi ), stheta, ctheta * cos( phi );
         // NB : opencv images use to be stored in BGR
         const dvec3 rgb( row_data[x * 3 + 2], row_data[x * 3 + 1], row_data[x * 3] );
         for ( int shi = 0; shi < nbShCoeffs; ++shi )
         {
            row_coeffs[shi] += weight * rgb * sh::EvalSH( shLM[shi].x, shLM[shi].y, dir );
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

   string imgLstFilename = parser.get<string>( "imgLst" );
   string dbFilename = parser.get<string>( "@db" );
   const int mode = parser.get<int>( "mode" );
   const int shOrder = parser.get<int>( "shOrder" );

   // create/open the database
   std::unique_ptr<leveldb::DB> dbPtr;
   {
      leveldb::DB* db;
      leveldb::Options dbOpts;
      dbOpts.create_if_missing = true;
      leveldb::Status dbStatus = leveldb::DB::Open( dbOpts, dbFilename, &db );
      if ( !dbStatus.ok() )
      {
         cerr << dbStatus.ToString() << endl;
         return -1;
      }
      dbPtr.reset( db );
   }

   // iterate over the list of files
   if ( mode == 0 )
   {
      Perf profiler;
      profiler.start();

      ImgFileLst lst( imgLstFilename.c_str() );
      leveldb::WriteOptions dbWriteOpts;
      for ( size_t lst_i = 0; lst_i < lst.size(); ++lst_i )
      {
         const string& imgPath = lst.get( lst_i );

         // read the image
         cv::Mat img = cv_utils::imread32FC3( imgPath );
         if ( !img.data ) continue;

         // compute the sh coefficients
         vector<dvec3> shCoeff;
         if ( !computeSphericalHarmonics( img, shCoeff, shOrder ) )
         {
            cerr << "Cannot compute SH for " << imgPath << endl;
            continue;
         }

         // write the result to the database
         dbPtr->Put(
             dbWriteOpts,
             imgPath,
             leveldb::Slice(
                 reinterpret_cast<const char*>( value_ptr( shCoeff[0] ) ),
                 sizeof( dvec3 ) * shCoeff.size() ) );
      }

      profiler.stop();
      cout << "Sh computation time -> " << profiler.getMs() << endl;
   }

   // iterate over the db
   if ( mode == 1 )
   {
      leveldb::ReadOptions dbReadOpts;
      std::unique_ptr<leveldb::Iterator> itPtr( dbPtr->NewIterator( dbReadOpts ) );
      const int dim = 3 * sh::GetCoefficientCount( shOrder );

      // compute the mean
      VectorXd shMeanCoeffs = VectorXd::Zero( dim );
      double z = 0.0;
      for ( itPtr->SeekToFirst(); itPtr->Valid(); itPtr->Next() )
      {
         if ( itPtr->value().size() < sizeof( double ) * dim )
            return false;
         else
         {
            z += 1.0;
            shMeanCoeffs += Map<VectorXd>(
                reinterpret_cast<double*>( const_cast<char*>( itPtr->value().data() ) ), dim );
         }
      }
      std::cout << z << endl;
      if ( z > 0.0 ) 
      {
         z = 1.0 / z;
         shMeanCoeffs *= z;
      }

      // compute the variance
      MatrixXd shCovCoeffs = MatrixXd::Zero( dim, dim );
      int test = 0;
      for ( itPtr->SeekToFirst(); itPtr->Valid(); itPtr->Next() )
      {
         if ( itPtr->value().size() < sizeof( double ) * dim )
            return false;
         else
         {
            test++;
            VectorXd shSampleCoeffs =
                Map<VectorXd>(
                    reinterpret_cast<double*>( const_cast<char*>( itPtr->value().data() ) ), dim ) -
                shMeanCoeffs;
            shCovCoeffs += shSampleCoeffs * shSampleCoeffs.adjoint() ;
         }
      }
      shCovCoeffs *= z;
      std::cout << test << endl;
      std::cout << shMeanCoeffs << endl;
      std::cout << shCovCoeffs << endl;
   }

   return ( 0 );
}
