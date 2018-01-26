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
#include <random>
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
    "{mode           |0           | 0=compute coeffs, 1=estimate coeffs statistics, 2=generate "
    "maps  }"
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

static double shCoeffsMean8[] = {
    1.62535,      1.59993,      1.52873,      -0.129034,    -0.229063,   -0.370292,   0.00808474,
    0.00664647,   0.00937933,   -0.000757816, -0.00102151,  -0.00121659, 0.000707051, 0.000757851,
    0.00084574,   0.00244401,   0.0023705,    -3.71057e-05, -0.00725334, -0.0203365,  -0.0511394,
    0.000540162,  0.000259493,  -3.54734e-05, -0.0141541,   -0.0365667,  -0.0914037,  -0.0337918,
    -0.0471127,   -0.0681103,   0.000125685,  0.000102473,  0.000561678, -0.0243074,  -0.0345659,
    -0.0506261,   0.00374988,   0.0020202,    0.00125083,   0.000341624, 0.000130869, -0.000197295,
    0.0064905,    0.0063412,    0.0062002,    -0.00141026,  -0.00159163, -0.001749,   0.000723703,
    0.00061244,   0.000979724,  -0.0014188,   -0.0013961,   -0.00209469, 0.00024993,  0.000391279,
    0.000524354,  -0.00097943,  0.000288477,  0.0018179,    -0.00940844, -0.0132097,  -0.0214507,
    -4.10496e-05, -6.45817e-05, -0.000133848, -0.0212887,   -0.0274884,  -0.0410339,  0.000122876,
    -3.21022e-05, -0.000388814, -0.0250338,   -0.032921,    -0.0499909,  -0.0142551,  -0.016832,
    -0.020492,    -0.000367205, -0.000425947, -0.000473871};
static double shCoeffsStd8[] = {
    0.429219,  0.429304,  0.501215,  0.322748,  0.292984,  0.33602,   0.144528,  0.144469,
    0.156821,  0.131678,  0.129134,  0.138005,  0.132425,  0.117658,  0.114917,  0.137179,
    0.125416,  0.127194,  0.139252,  0.134759,  0.142759,  0.0912928, 0.0881598, 0.0907393,
    0.190757,  0.183104,  0.196036,  0.119776,  0.116046,  0.135213,  0.0661479, 0.0619696,
    0.067106,  0.10324,   0.0980564, 0.11241,   0.075825,  0.0716308, 0.0735666, 0.0599346,
    0.0581499, 0.0613524, 0.0828133, 0.0766802, 0.0771773, 0.0881641, 0.0802417, 0.0787247,
    0.0601254, 0.0566595, 0.0619334, 0.0568282, 0.0544685, 0.0605963, 0.0476382, 0.0457157,
    0.0499598, 0.0587511, 0.0565128, 0.0624207, 0.0658961, 0.0646426, 0.0695578, 0.0473787,
    0.0467925, 0.0500343, 0.076179,  0.074809,  0.0810718, 0.0496845, 0.0469973, 0.0505273,
    0.0915342, 0.0895412, 0.0977202, 0.0627302, 0.0623561, 0.0720656, 0.0399477, 0.038154,
    0.0421067};

void generateSphericalHarmonics( vector<dvec3>& shCoeff, const int shOrder )
{
   static std::random_device rd;
   static std::mt19937 gen( rd() );
   std::uniform_real_distribution<> dis( -1.0, 1.0 );
   
   const int nbShCoeffs( sh::GetCoefficientCount( shOrder ) );
   shCoeff.resize( nbShCoeffs, dvec3( 0.0 ) );
   double* shCoeffPtr = glm::value_ptr(shCoeff[0]);
   for ( size_t i = 0 ; i < 3*nbShCoeffs; ++i )
   {
      std::normal_distribution<> dis{shCoeffsMean8[i],1.5};
      shCoeffPtr[i] = dis( gen );
   }
}

void renderEnvMapFromCoeff( Mat& envMap, const vector<dvec3>& shCoeff, const int shOrder )
{
   // convert input coeffs
   const int nbShCoeffs( sh::GetCoefficientCount( shOrder ) );
   vector<Array3f> shCoeffs( nbShCoeffs );
   for ( int shi = 0; shi < nbShCoeffs; ++shi )
   {
      shCoeffs[shi] << shCoeff[shi].r, shCoeff[shi].g, shCoeff[shi].b;
   }

#pragma omp parallel for
   for ( size_t y = 0; y < envMap.rows; y++ )
   {
      float* row_data = envMap.ptr<float>( y );
      const double theta = ( 0.5 - ( y + 0.5 ) / envMap.rows ) * M_PI;
      const double stheta = sin( theta );
      const double ctheta = cos( theta );

      for ( size_t x = 0; x < envMap.cols; x++ )
      {
         const double phi = ( 2.0 * ( x + 0.5 ) / envMap.cols - 1.0 ) * M_PI;
         Vector3d dir;
         dir << ctheta * sin( phi ), stheta, ctheta * cos( phi );

         Array3f irradiance = sh::EvalSHSum( shOrder, shCoeffs, dir );

         row_data[x * 3 + 2] = irradiance( 0 );
         row_data[x * 3 + 1] = irradiance( 1 );
         row_data[x * 3 + 0] = irradiance( 2 );
      }
   }
}

void printShCoeffs(const vector<dvec3>& shCoeff)
{
   for ( auto& sh : shCoeff )
   {
      cout << sh.r << " " << sh.g << " " << sh.b << " ";
   }
   cout << endl;
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
   if ( ( mode == 0 ) || ( mode == 3 ) )
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

         // testing
         if (mode == 3 )
         {
            Mat envMap32f( img.rows, img.cols, CV_32FC3 );
            renderEnvMapFromCoeff( envMap32f, shCoeff, shOrder );
            Mat envMap( img.rows, img.cols, CV_8UC3 );
            envMap32f.convertTo( envMap, CV_8U, 255.0 );
            printShCoeffs(shCoeff);
            imshow( "envMap", envMap );
            waitKey( 1 );
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

   // iterate over the list of files to create
   if ( mode == 2 )
   {
      const int envMapWidth( 1024 );
      const int envMapHeight( 512 );

      Perf profiler;
      profiler.start();

      ImgFileLst lst( imgLstFilename.c_str(), false );

      cout << "Read : " << imgLstFilename << " : " << lst.size() << endl;

      leveldb::WriteOptions dbWriteOpts;
      for ( size_t lst_i = 0; lst_i < lst.size(); ++lst_i )
      {
         const string& imgPath = lst.get( lst_i );

         cout << "Processing : " << imgPath << endl;

         // generate random sh coeffs
         vector<dvec3> shCoeff;
         generateSphericalHarmonics( shCoeff, shOrder );

         Mat envMap32f( envMapHeight, envMapWidth, CV_32FC3 );
         renderEnvMapFromCoeff( envMap32f, shCoeff, shOrder );
         Mat envMap( envMapHeight, envMapWidth, CV_8UC3 );
         envMap32f.convertTo( envMap, CV_8U * 255.0 );
         try
         {
            imwrite( imgPath.c_str(), envMap );
            imshow( "envMap", envMap32f * 255.0 );
            waitKey( 1 );
         }
         catch ( ... )
         {
            cout << "Error writing : " << imgPath << endl;
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
      cout << "Sh generation time -> " << profiler.getMs() << endl;
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
            shCovCoeffs += shSampleCoeffs * shSampleCoeffs.adjoint();
         }
      }
      shCovCoeffs *= z;
      std::cout << test << endl;
      std::cout << shMeanCoeffs << endl;
      std::cout << shCovCoeffs << endl;
   }

   return ( 0 );
}
