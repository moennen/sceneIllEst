/*!
 * *****************************************************************************
 *   \file splatFaceTextures.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-03-19
 *   *****************************************************************************/

#include "utils/imgFileLst.h"
#include "utils/cv_utils.h"

#include <librenderinterface/RI.h>
#include <librenderinterface/RIDevice.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <boost/filesystem.hpp>

#include <memory>
#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;
using namespace boost;

namespace
{
  inline void printCpError(const RIError& err)
  {
    cerr << "CPError : " << err.typeToStr( err.error ) << " "
         << err.msg << endl;
  }
}

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@faceImgLst    |         | face image list  }"
    "{@faceImgDir    |         | face image root directory  }"
    "{@outDir        |         | output directories   }"
    "{@outSize       |         | size of the output splatted texture }"
    "{@kernelPath    |         | path where the compute kernels are located}"
    "{show           |         |    }"
    "{nowrite        |         |    }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }

   const bool doShow = parser.get<bool>( "show" );
   const bool doWrite = !parser.get<bool>( "nowrite" );
   const filesystem::path outRootPath( parser.get<string>( "@outDir" ) );
   const filesystem::path cpKernelPath( parser.get<string>( "@kernelPath" ) );

   // face images dataset RGB / UVD / NORMALS
   ImgNFileLst imgLst(
       3,
       parser.get<string>( "@faceImgLst" ).c_str(),
       parser.get<string>( "@faceImgDir" ).c_str() );
   if ( imgLst.size() == 0 )
   {
      cerr << "Invalid face image list : " << parser.get<string>( "@faceImgLst" ) << endl;
      return -1;
   }

   // Compute ressources
   auto cpInstance = unique_ptr<RI>( RI::createRenderingInterface( RI::Type::Vulkan ) );
   auto cpError = cpInstance->initialize();
   if ( !cpError.ok() ) printCpError(cpError);
   
   auto cpDevice = cpInstance->createDevice( 0, cpKernelPath.string().c_str(), false );
   cerr << "The device's name is : " << cpDevice->name() << endl;
   // splat texture pipeline
   const RIComputePipeline* cpPipeline = nullptr;
   {
      RIPipelineLayout cpPipLayout;
      cpPipLayout.descriptors = {
          {RIDescriptorType::UNIFORM_BUFFER, RI_COMPUTE_STAGE, 1},   // Uniform buffer
          {RIDescriptorType::STORAGE_BUFFER, RI_COMPUTE_STAGE, 1},   // Input RGB Diffuse
          {RIDescriptorType::STORAGE_BUFFER, RI_COMPUTE_STAGE, 1},   // Input UVD
          {RIDescriptorType::STORAGE_BUFFER, RI_COMPUTE_STAGE, 1}};  // Output Splatted
      auto cpPipDesc = RIComputePipelineDescriptor( cpPipLayout, string( "/splatFaceTexture" ) );
      cpDevice->createComputePipelines( &cpPipDesc, 1 );
      auto errsCreatingPipelines = cpDevice->createComputePipelines( &cpPipDesc, 1 );
      for (const auto& err : errsCreatingPipelines) if ( !err.ok() ) printCpError(err);
      cpPipeline = cpDevice->acquireComputePipeline( cpPipDesc );
   }

   // out texture sizes
   const unsigned szOut = parser.get<unsigned>( "@outSize" );
   const uvec3 dimsOutBuffer( szOut, szOut, 3 );
   const size_t szOutBuffer = sizeof( float ) * dimsOutBuffer.x * dimsOutBuffer.y * dimsOutBuffer.z;

   //------------------------------------------------------------------------- Load model
   for ( int s = 0; s < imgLst.size(); ++s )
   {
      // Read input

      Mat matInRGB = cv_utils::imread32FC3( imgLst.filePath( s, 0 ), false, true );
      if ( matInRGB.empty() ) continue;

      const uvec3 dimsInBuffer( matInRGB.cols, matInRGB.rows, 3 );
      const size_t szInBuffer = sizeof( float ) * dimsInBuffer.x * dimsInBuffer.y * dimsInBuffer.z;

      // TODO UVS inferences
      // for now assume the input has the UVs

      Mat matInUVD = cv_utils::imread32FC3( imgLst.filePath( s, 1 ), false, false );

      auto cpInTransfer = cpDevice->createTransferCmdList( "InTransfer" );

      // Uniform buffer : contains only the buffer dimensions
      auto bufferDims = cpDevice->createBuffer( 2 * sizeof( uvec3 ), RI_USAGE_TRANSFER_DST );
      {
         auto staging = cpDevice->createStagingBuffer(
             2 * sizeof( uvec3 ), RI_USAGE_TRANSFER_SRC, "stagingDims" );
         memcpy( staging->data(), value_ptr( dimsInBuffer ), sizeof( uvec3 ) );
         memcpy(
             ( (char*)(staging->data()) + sizeof( uvec3 ) ),
             value_ptr( dimsOutBuffer ),
             sizeof( uvec3 ) );
         cpInTransfer->uploadToBuffer( staging, bufferDims );
      }

      // Input rgb buffer
      auto bufferInRGB = cpDevice->createBuffer( szInBuffer, RI_USAGE_TRANSFER_DST );
      {
         auto staging =
             cpDevice->createStagingBuffer( szInBuffer, RI_USAGE_TRANSFER_SRC, "stagingInRGB" );
         memcpy( staging->data(), matInRGB.data, szInBuffer );
         cpInTransfer->uploadToBuffer( staging, bufferInRGB );
      }

      // Input uvd buffer
      auto bufferInUVD = cpDevice->createBuffer( szInBuffer, RI_USAGE_TRANSFER_DST );
      {
         auto staging =
             cpDevice->createStagingBuffer( szInBuffer, RI_USAGE_TRANSFER_SRC, "stagingInUVD" );
         memcpy( staging->data(), matInUVD.data, szInBuffer );
         cpInTransfer->uploadToBuffer( staging, bufferInUVD );
      }

      cpInTransfer->submit();

      // Splat the texture

      auto cpSplatCmdList = cpDevice->createComputeCmdList( "SplatFaceTexture" );
      cpSplatCmdList->addWaitCommand( cpInTransfer, RI_TRANSFER_STAGE );
      cpSplatCmdList->setPipeline( cpPipeline );
      cpSplatCmdList->setBufferAt( bufferDims, 0, 0 );
      cpSplatCmdList->setBufferAt( bufferInRGB, 0, 1 );
      cpSplatCmdList->setBufferAt( bufferInUVD, 0, 2 );
      auto bufferOutRGB = cpDevice->createBuffer( szOutBuffer, RI_USAGE_TRANSFER_SRC );
      cpSplatCmdList->setBufferAt( bufferOutRGB, 0, 3 );
      RIComputeCmdList::DispatchInfo cpSplatDispatchInfos = {
          {( dimsInBuffer.x / 16u ) + std::min( 1u, dimsInBuffer.x % 16u ),
           ( dimsInBuffer.y / 16u ) + std::min( 1u, dimsInBuffer.y % 16u ),
           1}};
      cpSplatCmdList->dispatch( cpSplatDispatchInfos );
      cpSplatCmdList->submit();

      // Process the result
      auto cpOutTransfer = cpDevice->createTransferCmdList( "OutTransfer" );
      cpOutTransfer->addWaitCommand( cpSplatCmdList, RI_COMPUTE_STAGE );
      auto stagingOutRGB =
          cpDevice->createStagingBuffer( szOutBuffer, RI_USAGE_TRANSFER_DST, "stagingOutRGB" );
      cpOutTransfer->downloadFromBuffer( bufferOutRGB, stagingOutRGB );
      cpOutTransfer->submit();

      // Sync CPU/GPU
      cpOutTransfer->waitForCompletion();
      Mat matOutRGB( dimsOutBuffer.y, dimsOutBuffer.x, CV_32FC3 );
      memcpy( matOutRGB.data, stagingOutRGB->data(), szOutBuffer );

      char sampleId[16];
      sprintf( sampleId, "%08d_", s );
      const string outBasename( sampleId );
      const string outBasenameFull = ( outRootPath / filesystem::path( sampleId ) ).string();

      if ( doWrite )
      {
         imwrite( outBasenameFull + "c.exr", matInRGB * 255.0 );
         imwrite( outBasenameFull + "uvd.exr", matInUVD );
         imwrite( outBasenameFull + "diff.png", matOutRGB * 255.0 );
      }

      std::cout << outBasename + "c.png " << outBasename + "uvd.exr " << outBasename + "diff.png"
                << std::endl;

      if ( doShow )
      {
         imshow( "inRGB", matInRGB );
         imshow( "inUVD", matInUVD );
         imshow( "outRGB", matOutRGB );
         // imshow( "diffuse", out_diffuse );
         waitKey( 0 );
      }
   }

   return ( 0 );
}