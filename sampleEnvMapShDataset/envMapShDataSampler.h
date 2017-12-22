//*****************************************************************************/
//
// Filename envMapShDataSampler.h
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
#ifndef _SAMPLEENVMAPSHDATASET_ENVMAPSHDATASAMPLER_H
#define _SAMPLEENVMAPSHDATASET_ENVMAPSHDATASAMPLER_H

#include <leveldb/db.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_on_sphere.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <glm/glm.hpp>

#include <memory>
#include <string>

class EnvMapShDataSampler 
{
   // sh order
   const int _shOrder;
   const int _nbShCoeffs;

   // database
   std::unique_ptr<leveldb::DB> _dbPtr;
   leveldb::ReadOptions _dbOpts;

   // key hash for sampling keys
   std::vector<std::string> _keyHash;

   // random samplers
   boost::random::mt19937 _rng;
   boost::random::uniform_int_distribution<> _keyGen;
   boost::uniform_on_sphere<float> _unifSphere;
   boost::variate_generator<boost::random::mt19937&, boost::uniform_on_sphere<float> > _sphereGen;
   boost::random::uniform_real_distribution<> _fovGen;
   boost::random::uniform_real_distribution<> _rollGen;

public :
   EnvMapShDataSampler( int shOrder, leveldb::DB* db, int seed );
   virtual ~EnvMapShDataSampler();

   bool sample(
    float* /*imgData*/,
    const glm::uvec3 sz,
    float* /*shData*/,
    float* /*camData*/ );
};

#endif // _SAMPLEENVMAPSHDATASET_ENVMAPSHDATASAMPLER_H
