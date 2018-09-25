/*!
 * *****************************************************************************
 *   \file drawFaceBBMatte.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-09-25
 *   *****************************************************************************/

#include "utils/cv_utils.h"
#include "utils/imgFileLst.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <boost/filesystem.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;
using namespace boost;

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@faceImgBBLst  |         | list of face with bbox infos }"
    "{@faceImgDir    |         | output directories   }"
    "{@outDir        |         | output directories   }"
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

   // Load background images
   ImgNFileLst imgLst(
       5,
       parser.get<string>( "@faceImgBBLst" ).c_str(),
       parser.get<string>( "@faceImgDir" ).c_str() );
   if ( imgLst.size() == 0 )
   {
      cerr << "Invalid face image list : " << parser.get<string>( "@faceImgBBLst" ) << endl;
      return -1;
   }

   //--- write params
   const int iNbMaxPerGroup = 10000;
   unsigned uGroupId = 0;

   //--- list loop
   for ( int s = 0; s < imgLst.size(); ++s )
   {
      // load the inputs data
      Mat matFaceInput = cv_utils::imread32FC3( imgLst.filePath( s, 0 ) );
      if ( matFaceInput.empty() ) continue;
      vec2 v2PosFace;
      vec2 v2SzFace;
      try
      {
         v2PosFace.x = std::atof( imgLst( s, 1 ).c_str() );
         v2PosFace.y = std::atof( imgLst( s, 2 ).c_str() );
         v2SzFace.x = std::atof( imgLst( s, 3 ).c_str() );
         v2SzFace.y = std::atof( imgLst( s, 4 ).c_str() );
      }
      catch ( ... )
      {
         cerr << "Cannot parse sample : " << imgLst.filePath( s, 0 ) << imgLst( s, 1 ) << " "
              << imgLst( s, 2 ) << " " << imgLst( s, 3 ) << " " << imgLst( s, 4 ) << endl;
         continue;
      }

      // Create the matte output
      Mat matFaceMatte( matFaceInput.rows, matFaceInput.cols, CV_8UC1, Scalar( 0 ) );
      const vec2 v2FaceOff = v2SzFace * 0.17f;
      const array<Point, 4> arrFacePoly = {
          Point( v2PosFace.x + v2FaceOff.x, v2PosFace.y + v2FaceOff.y ),
          Point( v2PosFace.x + v2FaceOff.x, v2PosFace.y - v2FaceOff.y + v2SzFace.y ),
          Point( v2PosFace.x - v2FaceOff.x + v2SzFace.x, v2PosFace.y - v2FaceOff.y + v2SzFace.y ),
          Point( v2PosFace.x - v2FaceOff.x + v2SzFace.x, v2PosFace.y + v2FaceOff.y )};
      fillConvexPoly( matFaceMatte, &arrFacePoly[0], (int)arrFacePoly.size(), Scalar( 255 ) );

      if ( !doWrite )
      {
         rectangle( matFaceInput, arrFacePoly[0], arrFacePoly[2], Scalar( 0, 1.0, 0 ) );
      }

      // create the output directory if needed
      if ( ( s % iNbMaxPerGroup ) == 0 )
      {
         uGroupId += 1;
         char dirname[7];
         sprintf( dirname, "%06d", uGroupId );
         const filesystem::path outGroupRootPath =
             outRootPath / filesystem::path( std::string( dirname ) );
         if ( !filesystem::create_directory( outGroupRootPath ) )
         {
            cerr << "Cannot create directory : " << outGroupRootPath.string() << endl;
         }
      }

      // upload and write the maps
      char sampleId[16];
      sprintf( sampleId, "%06d/%08d_", uGroupId, s );
      const string outBasename( sampleId );
      const string outBasenameFull = ( outRootPath / filesystem::path( sampleId ) ).string();

      if ( doWrite )
      {
         imwrite( outBasenameFull + "c.png", matFaceInput * 255.0 );
         imwrite( outBasenameFull + "m.png", matFaceMatte );
      }

      std::cout << outBasename + "c.png " << outBasename + "m.png" << std::endl;

      if ( doShow )
      {
         imshow( "face", matFaceInput );
         imshow( "matte", matFaceMatte );
         waitKey( 0 );
      }
   }

   return ( 0 );
}