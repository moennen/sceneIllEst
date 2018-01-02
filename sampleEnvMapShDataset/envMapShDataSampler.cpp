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

#include "utils/cv_utils.h"
#include "sh/spherical_harmonics.h"

#include <iostream>
#include <Eigen/Dense>

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
   const double z = 0.5*sample.rows / std::tan(0.5*fov);

   // sample
#pragma omp parallel for
   for ( size_t y = 0; y < sample.rows; y++ )
   {
      const float* row_data = sample.ptr<float>( y );
      for ( size_t x = 0; x < sample.cols; x++ )
      {  
         Vector3d dir((double)x-0.5*sample.cols,(double)y-0.5*sample.rows,z);
         dir.normalize();
         dir = rotMat*dir;
         glm::vec2 pos( 
            envMap.cols*0.5*(atan2(dir[0], dir[2])/M_PI+1.0),
            envMap.rows*(asin(glm::clamp(dir[1], -1.0, 1.0))/M_PI+0.5) );
         sample.at<Vec3f>( y, x ) = cv_utils::imsample32FC3( envMap, pos );
      }
   }
}

bool renderEnvMapFromCoeff( Mat& envMap,  const int shOrder, const vector<Array3f>& shs)
{
// sample

#pragma omp parallel for
   for ( size_t y = 0; y < envMap.rows; y++ )
   {
      //const double theta = ( ( y + 0.5 ) / envMap.rows - 0.5 ) * M_PI;
      const double theta = sh::ImageYToTheta(y, envMap.rows);
      const double stheta = sin( theta );
      const double ctheta = cos( theta );
      float* row_data = envMap.ptr<float>( y );
      
      for ( size_t x = 0; x < envMap.cols; x++ )
      {
	 //const double phi = ( 2.0 * ( x + 0.5 ) / envMap.cols - 1.0 ) * M_PI;
         const double phi = sh::ImageXToPhi(x, envMap.cols);
         Vector3d dir;
         dir << sin( phi ), -stheta * cos( phi ), ctheta * cos( phi );
         dir.normalize();
         Array3f irradiance = sh::EvalSHSum(shOrder, shs, phi, theta); //sh::RenderDiffuseIrradiance(shs, dir);

	 row_data[x*3+2] = irradiance(0);
	 row_data[x*3+1] = irradiance(1);
	 row_data[x*3+0] = irradiance(2);
      }
   }
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
      _unifSphere( 3 ),
      _sphereGen( _rng, _unifSphere ),
      _fovGen( 10.0, 140.0 ),
      _rollGen(-180.0,180.0 )
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

EnvMapShDataSampler::~EnvMapShDataSampler(){};

bool EnvMapShDataSampler::sample(
    float* imgData,
    const uvec3 sz,
    float* shData,
    float* camData )
{

   // tests

   std::unique_ptr<leveldb::Iterator> itPtr( _dbPtr->NewIterator( _dbOpts ) );

   const size_t imgSize = sz.x*sz.y*3;

   for ( size_t s = 0; s < sz.z; ++s )
   {
      // sample the key
      const int keyId = _keyGen( _rng );
      // sample the point on a sphere
      const std::vector<float> camLookAt = _sphereGen();
      const double camRoll = M_PI*_rollGen( _rng )/180.0;
      // extract the pitch/yaw
      Vector3d rotAxis;
      rotAxis << camLookAt[0], camLookAt[1], camLookAt[2];
      rotAxis.normalize();
      const double camPitch =  asin(clamp(rotAxis[1],-1.0,1.0) );
      const double camYaw   =  atan2(rotAxis[0], rotAxis[2]);
      
      Matrix3d rot = AngleAxisd(camPitch,Vector3d::UnitX()).toRotationMatrix() *
                     AngleAxisd( camYaw , Vector3d::UnitY() ).toRotationMatrix() * 
                     AngleAxisd( camRoll, Vector3d::UnitZ() ).toRotationMatrix();
      Quaterniond quat( rot );

      // sample the fov in radians
      const float camFoV = 120.0 * M_PI / 180.0; //_fovGen( _rng ) * M_PI / 180.0;
      //cout << camFoV * 180.0 / M_PI << endl;

      float* camDataPtr = camData + s*nbCameraParams();
      camDataPtr[0] = camFoV;
      camDataPtr[1] = camPitch;	
      camDataPtr[2] = camYaw;	
      camDataPtr[3] = camRoll;	

      // Retrieve the sh coefficients
      itPtr->Seek( _keyHash[keyId] );
      if ( !itPtr->Valid() || itPtr->value().size() < sizeof( double ) * 3 * _nbShCoeffs )
         return false;
      vector<Array3f> shCoeffs( _nbShCoeffs );
      const double* value = reinterpret_cast<const double*>( itPtr->value().data() );
      float* shDataPtr = shData + s*3*_nbShCoeffs;
      for ( int shi = 0; shi < _nbShCoeffs; ++shi )
      {
	 shDataPtr[shi*3] = value[shi*3];
	 shDataPtr[shi*3+1] = value[shi*3+1];
	 shDataPtr[shi*3+2] = value[shi*3+2];

         Array3f sh;
         sh << value[shi * 3], value[shi * 3 + 1], value[shi * 3 + 2];
         shCoeffs[shi] = sh;
      }

      // Rotate the sh coefficients
      std::unique_ptr<sh::Rotation> shRot = sh::Rotation::Create( _shOrder, quat );
      shRot->Apply( shCoeffs, &shCoeffs );

      // Proto : create a map with the spherical harmonics
      //cvShImage img( sz.x, sz.y );
      //sh::RenderDiffuseIrradianceMap( shCoeffs, &img );
       
      cvShImage img( sz.y, 2*sz.y );
      renderEnvMapFromCoeff( img.cv(), _shOrder, shCoeffs );
      
      // Open the image

      Matrix3d rotMat =  rot;
      Mat oimg = cv_utils::imread32FC3( _keyHash[keyId] );
      threshold(oimg,oimg,1.0,1.0,THRESH_TRUNC);
      if ( oimg.data )
      {
         //Mat small( sz.y, 2*sz.y, CV_32FC3 );
         Mat crop( sz.y, sz.x, CV_32FC3 );
         sampleImageFromEnvMap( oimg, crop, camFoV, rotMat );
	 memcpy(imgData+s*imgSize,crop.ptr<float>(),sizeof(float)*imgSize);
         //resize(oimg,small,small.size());
         //imshow( "original", small );
         //imshow( "crop", crop );
      }
      //double minVal, maxVal;
      //minMaxLoc( img.cv(), &minVal, &maxVal );
      //img.cv() = ( img.cv() - minVal ) / ( maxVal - minVal );
      //imshow( "irradianceMap", img.cv() );
      //waitKey();
   }
}
