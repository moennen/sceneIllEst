//*****************************************************************************/
//
// Filename envMapShDataSampler.h
//
//*****************************************************************************/
#ifndef _SAMPLEENVMAPSHDATASET_ENVMAPSHDATASAMPLER_H
#define _SAMPLEENVMAPSHDATASET_ENVMAPSHDATASAMPLER_H

#include <leveldb/db.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_on_sphere.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

#include <glm/glm.hpp>
#include <Eigen/Dense>

#include <boost/filesystem.hpp>

#include <memory>
#include <string>

class EnvMapShDataSampler
{
   // color space linear import
   const bool _linearCS;

   // sh order
   const int _shOrder;
   const int _nbShCoeffs;

   // database
   std::unique_ptr<leveldb::DB> _dbPtr;
   leveldb::ReadOptions _dbOpts;

   // image location
   boost::filesystem::path _imgRootDir;

   // key hash for sampling keys
   std::vector<std::string> _keyHash;

   // random samplers
   boost::random::mt19937 _rng;
   boost::random::uniform_int_distribution<> _keyGen;
   boost::variate_generator<boost::random::mt19937&, boost::random::normal_distribution<> > _fovGen;
   boost::variate_generator<boost::random::mt19937&, boost::random::normal_distribution<> >
       _pitchGen;
   boost::variate_generator<boost::random::mt19937&, boost::random::normal_distribution<> >
       _rollGen;
   boost::random::uniform_real_distribution<> _yawGen;

   Eigen::VectorXd _shMeanCoeffs;
   Eigen::MatrixXd _shCovCoeffs;

  public:
   EnvMapShDataSampler( int shOrder, leveldb::DB* db, const std::string& imgRootDir, int seed, bool linearCS );
   virtual ~EnvMapShDataSampler();

   bool sample( float* /*imgData*/, const glm::uvec3 sz, float* /*shData*/, float* /*camData*/ );

   int nbShCoeffs() const { return _nbShCoeffs; }
   static int nbCameraParams() { return 4; }  // 3 rotations + fov

  public:

   static int nbShCoeffs( const int shOrder );
   static bool
   loadSampleImg( const char* fileName, float* /*imgBuff*/, const size_t w, const size_t h, const bool linearCS );
   static bool generateEnvMapFromShCoeffs(
       const int shOrder,
       const float* shCoeffs,
       float* envMap,
       const int w,
       const int h );
};

#endif  // _SAMPLEENVMAPSHDATASET_ENVMAPSHDATASAMPLER_H
