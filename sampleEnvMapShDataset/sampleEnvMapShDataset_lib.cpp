/*! *****************************************************************************
 *   \file sampleEnvMapShDataset_main.cpp
 *   \author moennen
 *   \brief
 *   \date 2017-12-20
 *   *****************************************************************************/

#include "utils/cv_utils.h"
#include "sh/spherical_harmonics.h"

#include <glm/glm.hpp>

#include <ctime>
#include <iostream>

#include <leveldb/db.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_on_sphere.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@db            |         | leveldb database     }";

class cvShImage : public sh::Image
{
   Mat _m;

  public:
   cvShImage( int w, int h ) : _m( w, h, CV_32FC3 ) {}

   int width() const { return _m.cols; }
   int height() const { return _m.rows; }

   Array3f GetPixel( int x, int y ) const
   {
      Vec3f rgb = _m.at<Vec3f>( y, x );
      Array3f pix;
      pix << rgb.val[2], rgb.val[1], rgb.val[0];
      return pix;
   }

   void SetPixel( int x, int y, const Array3f& v )
   {
      Vec3f rgb = {v( 2 ), v( 1 ), v( 0 )};
      _m.at<Vec3f>( y, x ) = rgb;
   }

   Mat& cv() { return _m; }
};

struct EnvMapShDataSampler
{
   // sh order
   const int _shOrder;
   const int _nbShCoeffs;

   // database
   std::unique_ptr<leveldb::DB> _dbPtr;
   leveldb::ReadOptions _dbOpts;

   // key hash for sampling keys
   std::vector<string> _keyHash;

   // random samplers
   boost::random::mt19937 _rng;
   boost::random::uniform_int_distribution<> _keyGen;
   boost::uniform_on_sphere<float> _unifSphere;
   boost::variate_generator<boost::random::mt19937&, boost::uniform_on_sphere<float> > _sphereGen;
   boost::random::uniform_real_distribution<> _fovGen;

   EnvMapShDataSampler( int shOrder, leveldb::DB* db, int seed )
       : _shOrder( shOrder ),
         _nbShCoeffs( sh::GetCoefficientCount( shOrder ) ),
         _dbPtr( db ),
         _rng( seed ),
         _unifSphere( 3 ),
         _sphereGen( _rng, _unifSphere ),
         _fovGen( 0.5, 179.5 )
   {
      // create the hash for sampling the keys
      std::unique_ptr<leveldb::Iterator> itPtr( _dbPtr->NewIterator( _dbOpts ) );
      _keyHash.reserve( 10000 );
      for ( itPtr->SeekToFirst(); itPtr->Valid(); itPtr->Next() )
      {
         _keyHash.push_back( itPtr->key().ToString() );
      }
      _keyGen = boost::random::uniform_int_distribution<>( 0, _keyHash.size() - 1 );
   }

   bool sample( float* /*imgData*/, const glm::uvec3 sz, float* /*shData*/, float* /*camData*/ )
   {
      std::unique_ptr<leveldb::Iterator> itPtr( _dbPtr->NewIterator( _dbOpts ) );

      for ( size_t s = 0; s < sz.z; ++s )
      {
         // sample the key
         const int keyId = _keyGen( _rng );
         // sample the point on a sphere
         const std::vector<float> camLookAt = _sphereGen();
         // extract the pitch/yaw
         Vector3d rotAxis;
         rotAxis << camLookAt[0], camLookAt[1], camLookAt[2];
         rotAxis.normalize();
         double phi, theta;
         sh::ToSphericalCoords(rotAxis, &phi, &theta);
         AngleAxisd rot = AngleAxisd( phi, Vector3d::UnitZ() ) /** 
                          AngleAxisd( theta, Vector3d::UnitY() )*/ ;
         Quaterniond quat( rot );
         
         // sample the fov in radians
         const float camFoV = _fovGen( _rng ) * M_PI / 180.0;

         // Retrieve the sh coefficients
         itPtr->Seek( _keyHash[keyId] );
         if ( !itPtr->Valid() || itPtr->value().size() < sizeof( double ) * 3 * _nbShCoeffs ) return false;
         std::vector<Array3f> shCoeffs( _nbShCoeffs );
         const double* value = reinterpret_cast<const double*>( itPtr->value().data() );
         for ( int shi = 0; shi < _nbShCoeffs; ++shi )
         {
            Array3f sh;
            sh << value[shi * 3], value[shi * 3 + 1], value[shi * 3 + 2];
            shCoeffs[shi] = sh;
         }

         // Rotate the sh coefficients
         std::unique_ptr<sh::Rotation> shRot = sh::Rotation::Create( _shOrder, quat );
         shRot->Apply( shCoeffs, &shCoeffs );

         // Proto : create a map with the spherical harmonics
         cvShImage img( sz.x, sz.y );
         sh::RenderDiffuseIrradianceMap( shCoeffs, &img );



         // Open the image
         /*Mat oimg = cv_utils::imread32FC3( _keyHash[keyId] );
         if (oimg.data)
         {
            Mat small(sz.x, sz.y, CV_32FC3);
            resize(oimg,small,small.size());
            imshow("original", small);
         }
         double minVal, maxVal;
         minMaxLoc(img.cv(), &minVal, &maxVal);
         cout << minVal << " - " << maxVal << endl;
         img.cv() = (img.cv() - minVal) / (maxVal-minVal);
         imshow( "irradianceMap", img.cv() );
         waitKey();*/
      }
   }
};

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }
   string outputFilename = parser.get<string>( "@db" );

   // open the database
   std::unique_ptr<EnvMapShDataSampler> shSampler;
   {
      leveldb::DB* db;
      leveldb::Options dbOpts;
      leveldb::Status dbStatus = leveldb::DB::Open( dbOpts, outputFilename, &db );
      if ( !dbStatus.ok() )
      {
         cerr << dbStatus.ToString() << endl;
         return -1;
      }
      shSampler.reset( new EnvMapShDataSampler( 7, db, std::time( 0 ) ) );
   }

   // sample
   glm::uvec3 sz( 256, 256, 100 );
   shSampler->sample( nullptr, sz, nullptr, nullptr );

   return ( 0 );
}
