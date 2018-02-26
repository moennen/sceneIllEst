//*****************************************************************************/
//
// Filename envMapShDataSampler.C
//
// Copyright (c) 2017 Autodesk, Inc.
// All rights reserved.
//
// This computer source code and related instructions and comments are the
// unpublished confidential and proprietary information of Autodesk, Inc.
// and are protected under applicable copyright and trade secret law.
// They may not be disclosed to, copied or used by any third party without
// the prior written consent of Autodesk, Inc.
//*****************************************************************************/

#include <sampleEnvMapShDataset/envMapShDataSampler.h>

#include "utils/perf.h"
#include "utils/cv_utils.h"
#include "sh/spherical_harmonics.h"

#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;
using namespace Eigen;

#define M_H_INV_PI 0.159154943
#define EPS 1e-09;

//-----------------------------------------------------------------------------
namespace
{
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

bool sampleImageFromEnvMap( Mat& envMap, Mat& sample, float fov, const Matrix3d& rotMat )
{
   const double z = 0.5 * sample.rows / std::tan( 0.5 * fov );

   // sample
#pragma omp parallel for
   for ( size_t y = 0; y < sample.rows; y++ )
   {
      const float* row_data = sample.ptr<float>( y );
      for ( size_t x = 0; x < sample.cols; x++ )
      {
         Vector3d dir( (double)x - 0.5 * sample.cols, (double)y - 0.5 * sample.rows, z );
         dir.normalize();
         dir = rotMat * dir;
         glm::vec2 pos(
             envMap.cols * 0.5 * ( atan2( dir[0], dir[2] ) / M_PI + 1.0 ),
             envMap.rows * ( asin( glm::clamp( dir[1], -1.0, 1.0 ) ) / M_PI + 0.5 ) );
         sample.at<Vec3f>( y, x ) = cv_utils::imsample32FC3( envMap, pos );
      }
   }
}

// integrate over an image to estimate the spherical harmonics of a scene
// TODO : change the assumption that the unseen scene part is black
bool estimateSphericalHarmonics( Mat& sample, float fov, vector<dvec3> shCoeff, int shOrder = 4 )
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
   const double pixel_area = ( fov / sample.rows ) * ( fov / sample.rows );
   shCoeff.resize( nbShCoeffs, dvec3( 0.0 ) );
   vector<vector<dvec3> > img_coeffs( sample.rows, shCoeff );

   const double z = 0.5 * sample.rows / std::tan( 0.5 * fov );

   // sample
#pragma omp parallel for
   for ( size_t y = 0; y < sample.rows; y++ )
   {
      const float* row_data = sample.ptr<float>( y );
      vector<dvec3>& row_coeffs = img_coeffs[y];
      for ( size_t x = 0; x < sample.cols; x++ )
      {
         Vector3d dir( (double)x - 0.5 * sample.cols, (double)y - 0.5 * sample.rows, z );
         dir.normalize();

         // compute sin(theta)
         // NB dir = (sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta))
         const double stheta = std::sqrt( dir[0] * dir[0] + dir[1] * dir[1] );
         const double weight = pixel_area * stheta;

         const dvec3 rgb( row_data[x * 3 + 2], row_data[x * 3 + 1], row_data[x * 3] );
         for ( int shi = 0; shi < nbShCoeffs; ++shi )
         {
            row_coeffs[shi] += weight * rgb * sh::EvalSH( shLM[shi].x, shLM[shi].y, dir );
         }
      }
   }

   // set the output
   for ( size_t y = 0; y < sample.rows; y++ )
   {
      const vector<dvec3>& row_coeffs = img_coeffs[y];
      for ( int shi = 0; shi < nbShCoeffs; ++shi )
      {
         shCoeff[shi] += row_coeffs[shi];
      }
   }

   return true;
}

void renderEnvMapFromCoeff( Mat& envMap, const int shOrder, const vector<Array3f>& shs )
{
   // sample

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

         Array3f irradiance = sh::EvalSHSum( shOrder, shs, dir );

         row_data[x * 3 + 2] = irradiance( 0 );
         row_data[x * 3 + 1] = irradiance( 1 );
         row_data[x * 3 + 0] = irradiance( 2 );
      }
   }
}

void copyToPlanarBuffer( float* buff, const Mat& img )
{
   const size_t imgSz = img.rows * img.cols;
   vector<Mat> img_split( img.channels() );
   split( img, &img_split[0] );
   for ( size_t c = 0; c < img.channels(); ++c )
   {
      memcpy( buff + c * imgSz, img_split[c].ptr<float>(), sizeof( float ) * imgSz );
   }
}

void copyToInterleavedBuffer( float* buff, const Mat& img )
{
   const size_t imgSz = img.rows * img.cols * img.channels();
   memcpy( buff, img.ptr<float>(), sizeof( float ) * imgSz );
}
}

//-----------------------------------------------------------------------------
//
//

EnvMapShDataSampler::EnvMapShDataSampler( int shOrder, leveldb::DB* db, int seed )
    : _shOrder( shOrder ),
      _nbShCoeffs( sh::GetCoefficientCount( shOrder ) ),
      _dbPtr( db ),
      _rng( seed ),
      _fovGen( _rng, boost::normal_distribution<>( 40.0, 20.0 ) ),
      _rollGen( _rng, boost::normal_distribution<>( 0.0, 15.0 ) ),
      _pitchGen( _rng, boost::normal_distribution<>( 0.0, 15.0 ) ),
      _yawGen( -180.0, 180.0 )
{
   // create the hash for sampling the keys
   std::unique_ptr<leveldb::Iterator> itPtr( _dbPtr->NewIterator( _dbOpts ) );
   _keyHash.reserve( 10000 );
   for ( itPtr->SeekToFirst(); itPtr->Valid(); itPtr->Next() )
   {
      _keyHash.push_back( itPtr->key().ToString() );
      // if ( _keyHash.size() == 10 ) break;
   }
   // cout << "WARNING : DS SIZE == " << _keyHash.size() << endl;
   _keyGen = boost::random::uniform_int_distribution<>( 0, _keyHash.size() - 1 );

   // compute mean / cov
   const int dim = 3 * _nbShCoeffs;

   // --> the mean
   _shMeanCoeffs = VectorXd::Zero( dim );
   double zMean = 0.0;
   for ( itPtr->SeekToFirst(); itPtr->Valid(); itPtr->Next() )
   {
      if ( itPtr->value().size() < sizeof( double ) * dim ) continue;
      zMean += 1.0;
      _shMeanCoeffs += Map<VectorXd>(
          reinterpret_cast<double*>( const_cast<char*>( itPtr->value().data() ) ), dim );
   }
   if ( zMean > 0.0 ) _shMeanCoeffs *= ( 1.0 / zMean );

   // compute the variance
   _shCovCoeffs = MatrixXd::Zero( dim, dim );
   double zCov = 0.0;
   for ( itPtr->SeekToFirst(); itPtr->Valid(); itPtr->Next() )
   {
      if ( itPtr->value().size() < sizeof( double ) * dim ) continue;

      zCov += 1.0;
      VectorXd shSampleCoeffs =
          Map<VectorXd>(
              reinterpret_cast<double*>( const_cast<char*>( itPtr->value().data() ) ), dim ) -
          _shMeanCoeffs;
      _shCovCoeffs += shSampleCoeffs * shSampleCoeffs.adjoint();
   }
   if ( zCov > 1.0 ) _shCovCoeffs *= ( 1.0 / ( zCov - 1.0 ) );

   //
   cout << "[ ";
   for ( int shi = 0; shi < _nbShCoeffs; ++shi )
   {
      cout << _shMeanCoeffs[shi] << ", ";
   }
   cout << " ]" << endl;

   cout << "[ ";
   for ( int shi = 0; shi < _nbShCoeffs; ++shi )
   {
      cout << sqrt(_shCovCoeffs(shi,shi)) << ", " ;
      //cout << "0.5, ";
   }
   cout << " ]" << endl;
}

EnvMapShDataSampler::~EnvMapShDataSampler(){};

bool EnvMapShDataSampler::sample( float* imgData, const uvec3 sz, float* shData, float* camData )
{
   // Perf profiler;
   // profiler.start();

   // tests
   std::unique_ptr<leveldb::Iterator> itPtr( _dbPtr->NewIterator( _dbOpts ) );

   const size_t imgSize = sz.x * sz.y * 3;

   for ( size_t s = 0; s < sz.z; ++s )
   {
      // sample the key
      const int keyId = _keyGen( _rng );
      // sample the point on a sphere

      double camRoll = 0.0; //_rollGen();
      while ( ( camRoll < -90.0 ) || ( camRoll > 90.0 ) ) camRoll = _rollGen();
      camRoll *= M_PI / 180.0;
      double camPitch = 0.0; //_pitchGen();
      while ( ( camPitch < -90.0 ) || ( camPitch > 90.0 ) ) camPitch = _pitchGen();
      camPitch *= M_PI / 180.0;
      const double camYaw = M_PI * _yawGen( _rng ) / 180.0;

      Matrix3d rot = AngleAxisd( camYaw, Vector3d::UnitY() ).toRotationMatrix();
      rot = AngleAxisd( camPitch, rot * Vector3d::UnitX() ).toRotationMatrix() * rot;
      rot = AngleAxisd( camRoll, rot * Vector3d::UnitZ() ).toRotationMatrix() * rot;
      Quaterniond quat( rot );

      // sample the fov in radians
      float camFoV = 90.0; //_fovGen();
      while ( ( camFoV < 20.0 ) || ( camFoV > 120.0 ) ) camFoV = _fovGen();
      camFoV *= M_PI / 180.0;

      float* camDataPtr = camData + s * nbCameraParams();
      camDataPtr[0] = camFoV;
      camDataPtr[1] = camPitch;
      camDataPtr[2] = camYaw;
      camDataPtr[3] = camRoll;

      // Retrieve the sh coefficients
      itPtr->Seek( _keyHash[keyId] );
      if ( !itPtr->Valid() || itPtr->value().size() < sizeof( double ) * 3 * _nbShCoeffs )
         return false;
      vector<Array3f> shCoeffs( _nbShCoeffs );

      VectorXd shSampleCoeffs = Map<VectorXd>(
          reinterpret_cast<double*>( const_cast<char*>( itPtr->value().data() ) ),
          3 * _nbShCoeffs );

      /*shSampleCoeffs = (shSampleCoeffs - _shMeanCoeffs);
      for ( int shi = 0; shi < 3 * _nbShCoeffs; ++shi )
      {
         shSampleCoeffs[shi] = shSampleCoeffs[shi] / sqrt(_shCovCoeffs(shi,shi));
      }*/

      float* shDataPtr = shData + s * 3 * _nbShCoeffs;
      for ( int shi = 0; shi < _nbShCoeffs; ++shi )
      {
         shDataPtr[shi * 3] = shSampleCoeffs[shi * 3];
         shDataPtr[shi * 3 + 1] = shSampleCoeffs[shi * 3 + 1];
         shDataPtr[shi * 3 + 2] = shSampleCoeffs[shi * 3 + 2];

         shCoeffs[shi] << shDataPtr[shi * 3], shDataPtr[shi * 3 + 1], shDataPtr[shi * 3 + 2];
      }

      // Rotate the sh coefficients
      std::unique_ptr<sh::Rotation> shRot = sh::Rotation::Create( _shOrder, quat );
      shRot->Apply( shCoeffs, &shCoeffs );

      // Proto : create a map with the spherical harmonics
      // cvShImage img( sz.x, sz.y );
      // sh::RenderDiffuseIrradianceMap( shCoeffs, &img );

      // cvShImage img( sz.y, 2 * sz.y );
      // renderEnvMapFromCoeff( img.cv(), _shOrder, shCoeffs );

      // Open the image

      Matrix3d rotMat = rot;
      Mat oimg = cv_utils::imread32FC3( _keyHash[keyId] );
      // std::cout << "Sampling image : " <<  _keyHash[keyId] << std::endl;
      // threshold( oimg, oimg, 1.0, 1.0, THRESH_TRUNC );
      if ( oimg.data )
      {
         // Mat small( sz.y, 2*sz.y, CV_32FC3 );
         Mat crop( sz.y, sz.x, CV_32FC3 );
         sampleImageFromEnvMap( oimg, crop, camFoV, rotMat );
         // DEBUG !!!!!!!!!!!!!!
         //resize(oimg,crop,crop.size());
         // resize(oimg,small,small.size());
         // imshow( "original", small );
         // imshow( "crop", crop );
         cvtColor( crop, crop, COLOR_BGR2RGB );
         copyToInterleavedBuffer( imgData + s * imgSize, crop );
      }
      else return false;
      // double minVal, maxVal;
      // minMaxLoc( img.cv(), &minVal, &maxVal );
      // img.cv() = ( img.cv() - minVal ) / ( maxVal - minVal );
      // imshow( "irradianceMap", img.cv() );
      // waitKey();
   }

   // profiler.stop();
   // cout << "SampleSh computation time -> " << profiler.getMs() << endl;
   return true;
}

int EnvMapShDataSampler::nbShCoeffs( const int shOrder )
{
   return sh::GetCoefficientCount( shOrder );
}

bool EnvMapShDataSampler::loadSampleImg(
    const char* fileName,
    float* imgData,
    const size_t w,
    const size_t h )
{
   Mat oimg = cv_utils::imread32FC3( fileName );
   if ( oimg.data )
   {
      Mat small( h, w, CV_32FC3 );
      resize( oimg, small, small.size(), 0.0, 0.0, INTER_AREA );
      waitKey();
      copyToInterleavedBuffer( imgData, small );

      return true;
   }

   return false;
}

bool EnvMapShDataSampler::generateEnvMapFromShCoeffs(
   const int shOrder,
   const float* shData,
   float* envMapData,
   const int w,
   const int h 
)
{
   // convert input coeffs
   const int nCoeffs(nbShCoeffs(shOrder));
   vector<Array3f> shCoeffs( nCoeffs );
   for ( int shi = 0; shi < nCoeffs; ++shi )
   {
      shCoeffs[shi] << shData[shi * 3], shData[shi * 3 + 1], shData[shi * 3 + 2];
   }
   cvShImage img( h, w );
   renderEnvMapFromCoeff( img.cv(), shOrder, shCoeffs );
   
   if ( img.cv().data )
   {
      cvtColor( img.cv(), img.cv(), COLOR_BGR2RGB );
      copyToInterleavedBuffer( envMapData, img.cv() );
      return true;
   }

   return false;
}
