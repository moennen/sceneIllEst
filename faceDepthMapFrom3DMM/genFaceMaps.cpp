/*!
 * *****************************************************************************
 *   \file genFaceMaps.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-09-19
 *   *****************************************************************************/

#include "utils/cv_utils.h"
#include "utils/imgFileLst.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "libImg2ImgCNNInfEng/i2iCNNEngFactory.h"

#include <boost/filesystem.hpp>

#include <random>
#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;
using namespace boost;

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@faceMapsModel |         | face maps model      }"
    "{@faceImgLst    |         | face image list      }"
    "{@outDir        |         | output directories   }"
    "{faceDiscModel  |''       | face disc model      }"
    "{discValThres   |0.0      |    }"
    "{inCHW          |false    |    }"
    "{forceSize      |-1       |    }"
    "{gpu            |         |    }"
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

   const bool gpu = parser.get<bool>( "gpu" );
   const bool doShow = parser.get<bool>( "show" );
   const bool doWrite = !parser.get<bool>( "nowrite" );
   const bool inCHW = parser.get<bool>( "inCHW" );
   const int forceSize = parser.get<int>( "forceSize" );
   const float discValThres = parser.get<float>( "discValThres" );

   const filesystem::path outRootPath( parser.get<string>( "@outDir" ) );

   // Load background images
   ImgNFileLst imgLst( 1, parser.get<string>( "@faceImgLst" ).c_str(), "" );
   if ( imgLst.size() == 0 )
   {
      cerr << "Invalid face image list : " << parser.get<string>( "@faceImgLst" ) << endl;
      return -1;
   }

   //--- create the face maps inference engine
   const string mapsModelName = parser.get<string>( "@faceMapsModel" );
   const string mapsInputName = {"adsk_Front"};
   const array<string, 2> mapsOutputNames = {"adsk_UVD", "adsk_Normals"};
   const string discModelName = parser.get<string>( "faceDiscModel" );
   unique_ptr<I2ICNNEng> cnnMapsEngine(
       I2ICNNEngFactory::createCNNEngine( I2ICNNEng::CNNModelFormat::TENSORFLOW_PB ) );
   const string discOutputName = {"adsk_SynthFace"};
   unique_ptr<I2ICNNEng> cnnDiscEngine(
       discOutputName.empty()
           ? nullptr
           : I2ICNNEngFactory::createCNNEngine( I2ICNNEng::CNNModelFormat::TENSORFLOW_PB ) );

   //---
   ivec3 input_sz( ivec3( -1 ) );
   const int nbOutputs = mapsOutputNames.size();
   vector<Mat> outputs( nbOutputs );
   vector<float*> outputs_data( nbOutputs );
   vector<ivec3> outputs_sz( nbOutputs, ivec3( -1 ) );
   Mat disc_output;
   float *disc_output_data=nullptr;

   //--- inference loop
   for ( int s = 0; s < imgLst.size(); ++s )
   {
      // load the inputs data
      Mat input = cv_utils::imread32FC3( imgLst.filePath( s, 0 ), false, true );
      if ( input.empty() ) continue;
      if ( forceSize > 0 ) cv_utils::resizeTo( input, uvec2( forceSize, forceSize ) );
      if ( inCHW ) cv_utils::toCHW32F( input );
      const ivec3 isz( input.cols, input.rows, 3 );
      if ( !all( equal( isz, input_sz ) ) )
      {
         input_sz = isz;
         if ( !cnnMapsEngine->setFromFile(
                  I2ICNNEng::CNNModelFormat::TENSORFLOW_PB,
                  mapsModelName.c_str(),
                  1,
                  &mapsInputName,
                  &input_sz,
                  nbOutputs,
                  &mapsOutputNames[0] ) )
         {
            cerr << "Cannot load the model : " << mapsModelName << endl;
            return -1;
         }
         cnnMapsEngine->printAllOperations();

         for ( int i = 0; i < nbOutputs; ++i )
         {
            outputs_sz[i] = cnnMapsEngine->getOutputsShape( i );
            outputs[i] = Mat( outputs_sz[i].y, outputs_sz[i].x, CV_32FC( outputs_sz[i].z ) );
            outputs_data[i] = reinterpret_cast<float*>( outputs[i].data );
         }

         if ( cnnDiscEngine.get() )
         {
            if ( !cnnDiscEngine->setFromFile(
                     I2ICNNEng::CNNModelFormat::TENSORFLOW_PB,
                     discModelName.c_str(),
                     nbOutputs,
                     &mapsOutputNames[0],
                     &outputs_sz[0],
                     1,
                     &discOutputName ) )
            {
               cerr << "Cannot load the model : " << discModelName << endl;
               return -1;
            }
            cnnDiscEngine->printAllOperations();

            const ivec3 disc_output_sz = cnnDiscEngine->getOutputsShape( 0 );
            disc_output = Mat( disc_output_sz.y, disc_output_sz.x, CV_32FC( disc_output_sz.z ) );
            disc_output_data = (float*)disc_output.data;
         }
      }

      // perform the maps inference
      const float *input_data = (float*)input.data;
      if ( !cnnMapsEngine->run( &input_data, &outputs_data[0] ) ) continue;

      // perform the disc inference
      float discVal = 1.0;
      if ( cnnDiscEngine.get() && cnnDiscEngine->run( (const float **)&outputs_data[0], &disc_output_data ) )
      {
        discVal = mean(disc_output)[0];
      }

      // upload and write the maps
      char sampleId[16];
      sprintf( sampleId, "%08d_", s );
      const string outBasename( sampleId );
      const string outBasenameFull = ( outRootPath / filesystem::path( sampleId ) ).string();

      if ( doWrite && (discVal >= discValThres) )
      {
         imwrite( outBasenameFull + "c.png", input*255.0 );
         imwrite( outBasenameFull + "uvd.exr", outputs[0] );
         imwrite( outBasenameFull + "n.exr", outputs[1] );
      }

      if (discVal >= discValThres)
      {
        std::cout << outBasename + "c.png " << outBasename + "uvd.exr " << outBasename + "n.exr"
                  << std::endl;
      }
      else
      {
        std::cerr << "Filtered face with value = " << discVal << " / " << discValThres << endl;
      }

      if ( doShow )
      {

         imshow( "color", input );
         imshow( "uvdepth", outputs[0]  );
         imshow( "normals", outputs[1] );
         waitKey( 0 );
      }
   }

   return ( 0 );
}