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
inline void printCpError( const RIError& err )
{
   cerr << "CPError : " << err.typeToStr( err.error ) << " " << err.msg << endl;
}

const int uNbTreadsPerBlockDim = 16u;
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
   if ( !cpError.ok() ) printCpError( cpError );

   auto cpDevice = cpInstance->createDevice( 0, cpKernelPath.string().c_str(), false );
   cerr << "The device's name is : " << cpDevice->name() << endl;
   // splat texture pipeline
   const RIComputePipeline* cpPipeline = nullptr;
   {
      RIPipelineLayout cpPipLayout;
      cpPipLayout.descriptors = {
          {RIDescriptorType::UNIFORM_BUFFER, RI_COMPUTE_STAGE, 1},   // Uniform buffer
          {RIDescriptorType::SAMPLED_TEXTURE, RI_COMPUTE_STAGE, 1},   // Input RGB Diffuse
          {RIDescriptorType::SAMPLED_TEXTURE, RI_COMPUTE_STAGE, 1},   // Input UVD
          {RIDescriptorType::STORAGE_BUFFER, RI_COMPUTE_STAGE, 1}};  // Output Splatted
      auto cpPipDesc = RIComputePipelineDescriptor( cpPipLayout, string( "/splatFaceTexture" ) );
      cpDevice->createComputePipelines( &cpPipDesc, 1 );
      auto errsCreatingPipelines = cpDevice->createComputePipelines( &cpPipDesc, 1 );
      for ( const auto& err : errsCreatingPipelines )
         if ( !err.ok() ) printCpError( err );
      cpPipeline = cpDevice->acquireComputePipeline( cpPipDesc );
   }
   // sampler
   const RISampler* riSamplerNearest = nullptr;
   {
      RISamplerDescriptor samplerDesc(
          RISamplerDescriptor::NEAREST,
          RISamplerDescriptor::NEAREST,
          RISamplerDescriptor::NEAREST,
          RISamplerDescriptor::CLAMP_TO_EDGE,
          RISamplerDescriptor::CLAMP_TO_EDGE );
      auto errsCreatingSampler = cpDevice->createSamplers( &samplerDesc, 1 );
      for ( const auto& err : errsCreatingSampler )
         if ( !err.ok() ) printCpError( err );
      riSamplerNearest = cpDevice->acquireSampler( samplerDesc );
   }

   // out texture sizes
   const unsigned szOut = parser.get<unsigned>( "@outSize" );
   const uvec3 dimsOutBuffer( szOut, szOut, 3 );
   const size_t szOutBuffer = sizeof( float ) * dimsOutBuffer.x * dimsOutBuffer.y * dimsOutBuffer.z;

   //------------------------------------------------------------------------- Load model
   for ( int s = 0; s < imgLst.size(); ++s )
   {
      // Read input

      Mat matInBGRA = cv_utils::imread32FC4( imgLst.filePath( s, 0 ), false, false );
      if ( matInBGRA.empty() ) continue;

      const uvec3 dimsInBuffer( matInBGRA.cols, matInBGRA.rows, 3 );
      const size_t szInBuffer = sizeof( float ) * dimsInBuffer.x * dimsInBuffer.y * dimsInBuffer.z;

      Mat matInDVU0 = cv_utils::imread32FC4( imgLst.filePath( s, 1 ), false, false );
      if ( matInDVU0.empty() ) continue;

      auto cpInTransfer = cpDevice->createTransferCmdList( "InTransferUniform" );
      auto cpSplatCmdList = cpDevice->createComputeCmdList( "SplatFaceTexture" );

      // Uniform buffer : contains only the buffer dimensions
      auto bufferDims = cpDevice->createBuffer( 2 * sizeof( uvec4 ), RI_USAGE_TRANSFER_DST );
      {
         auto staging = cpDevice->createStagingBuffer(
             2 * sizeof( uvec4 ), RI_USAGE_TRANSFER_SRC, "stagingDims" );
         uvec4 tmp = uvec4( dimsInBuffer, 0.0 );
         memcpy( staging->data(), value_ptr( tmp ), sizeof( uvec4 ) );
         tmp = uvec4( dimsOutBuffer, 0.0 );
         memcpy(
             ( (char*)( staging->data() ) + sizeof( uvec4 ) ), value_ptr( tmp ), sizeof( uvec4 ) );

         cpInTransfer->uploadToBuffer( staging, bufferDims );
      }

      // Input rgb buffer
      RITextureDescriptor texDescInBGRA{dimsInBuffer.x,
                                        dimsInBuffer.y,
                                        PF_128_4x32_FP,
                                        1,
                                        1,
                                        RI_USAGE_SHADER_READ | RI_USAGE_TRANSFER_DST};
      RITexture::RefPtr texInBGRA = cpDevice->createTexture2D( texDescInBGRA, "InBGRA" );
      {
         auto staging =
             cpDevice->createStagingBuffer( szInBuffer, RI_USAGE_TRANSFER_SRC, "stagingInBGRA" );
         memcpy( staging->data(), matInBGRA.data, szInBuffer );
         cpInTransfer->uploadToTexture( staging, texInBGRA );
      }

      // Input uvd buffer
      RITexture::RefPtr texInDVU0 = cpDevice->createTexture2D( texDescInBGRA, "InDUV0" );
      {
         auto staging =
             cpDevice->createStagingBuffer( szInBuffer, RI_USAGE_TRANSFER_SRC, "stagingInDUV0" );
         memcpy( staging->data(), matInDVU0.data, szInBuffer );
         cpInTransfer->uploadToTexture( staging, texInBGRA );
      }

      cpInTransfer->submit();
      cpSplatCmdList->addWaitCommand( cpInTransfer, RI_TRANSFER_STAGE );

      // Splat the texture
      cpSplatCmdList->setPipeline( cpPipeline );
      cpSplatCmdList->setBufferAt( bufferDims, 0, 0 );
      cpSplatCmdList->setTextureAt( texInBGRA, riSamplerNearest, 1 );
      cpSplatCmdList->setTextureAt( texInDVU0, riSamplerNearest, 2 );
      auto bufferOutBGR = cpDevice->createBuffer( szOutBuffer, RI_USAGE_TRANSFER_SRC );
      cpSplatCmdList->setBufferAt( bufferOutBGR, 0, 3 );
      RIComputeCmdList::DispatchInfo cpSplatDispatchInfos = {
          {( dimsInBuffer.x + uNbTreadsPerBlockDim - 1 ) / uNbTreadsPerBlockDim,
           ( dimsInBuffer.y + uNbTreadsPerBlockDim - 1 ) / uNbTreadsPerBlockDim,
           1}};
      cpSplatCmdList->dispatch( cpSplatDispatchInfos );
      cpSplatCmdList->submit();

      // Process the result
      auto cpOutTransfer = cpDevice->createTransferCmdList( "OutTransfer" );
      cpOutTransfer->addWaitCommand( cpSplatCmdList, RI_COMPUTE_STAGE );
      auto stagingOutBGR =
          cpDevice->createStagingBuffer( szOutBuffer, RI_USAGE_TRANSFER_DST, "stagingOutRGB" );
      cpOutTransfer->downloadFromBuffer( bufferOutBGR, stagingOutBGR );
      cpOutTransfer->submit();

      // Sync CPU/GPU
      cpOutTransfer->waitForCompletion();
      Mat matOutBGR( dimsOutBuffer.y, dimsOutBuffer.x, CV_32FC3 );
      memcpy( matOutBGR.data, stagingOutBGR->data(), szOutBuffer );

      char sampleId[16];
      sprintf( sampleId, "%08d_", s );
      const string outBasename( sampleId );
      const string outBasenameFull = ( outRootPath / filesystem::path( sampleId ) ).string();

      Mat matInBGR;
      cvtColor( matInBGRA, matInBGR, COLOR_BGR2BGRA );
      Mat matInDVU;
      cvtColor( matInDVU0, matInDVU, COLOR_BGR2BGRA );

      if ( doWrite )
      {
         imwrite( outBasenameFull + "c.exr", matInBGR * 255.0 );
         imwrite( outBasenameFull + "uvd.exr", matInDVU );
         imwrite( outBasenameFull + "diff.png", matOutBGR * 255.0 );
      }

      std::cout << outBasename + "c.png " << outBasename + "uvd.exr " << outBasename + "diff.png"
                << std::endl;

      if ( doShow )
      {
         imshow( "inRGB", matInBGR );
         imshow( "inUVD", matInDVU );
         imshow( "outRGB", matOutBGR );
         // imshow( "diffuse", out_diffuse );
         waitKey( 0 );
      }
   }

   return ( 0 );
}