/*!
 * *****************************************************************************
 *   \file main.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-02-19
 *   *****************************************************************************/

#include "flutils/imgFileLst.h"
#include "flutils/cv_utils.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <boost/filesystem.hpp>

#include <array>
#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;
using namespace boost;
using namespace Eigen;

namespace
{
//
bool filterEq( const Mat& img, const vec3& val )
{
   const ivec2 sz = {img.cols, img.rows};

   // store parallel results for every rows
   vector<bool> res( sz.y, false );
#pragma omp parallel for
   for ( unsigned y = 0; y < sz.y; y++ )
   {
      const vec3* ptr = img.ptr<vec3>( y );
      for ( unsigned x = 0; x < sz.x; x++ )
      {
         if ( all( equal( val, ptr[x] ) ) )
         {
            res[y] = true;
            break;
         }
      }
   }

   // check the results
   for ( const auto r : res )
      if ( r ) return true;
   return false;
}

}  // namespace

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@imgFileLst   |         | images list   }"
    "{@imgRootDir   |         | images root dir   }"
    "{@value        | 0.0     | value to find }"
    "{lstSz         | 1       |    }"
    "{lstId         | 0       |    }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }

   const int lstSz = parser.get<int>( "lstSz" );
   const int lstId = parser.get<int>( "lstId" );

   const vec3 value( parser.get<float>( "@value" ) );

   // Create the list
   ImgNFileLst imgLst(
       lstSz,
       parser.get<string>( "@imgFileLst" ).c_str(),
       parser.get<string>( "@imgRootDir" ).c_str() );
   if ( imgLst.size() == 0 )
   {
      cerr << "Invalid dataset : " << parser.get<string>( "@imgFileLst" ) << endl;
      return -1;
   }

   // stats
   const size_t nbSamples = imgLst.size();
   size_t nbFilteredSamples = 0;
   for ( size_t i = 0; i < nbSamples; ++i )
   {
      Mat img = cv_utils::imread32FC3( imgLst.filePath( i, lstId ), false, true );
      if ( img.empty() )
      {
         cout << imgLst.filePath( i, lstId ) << 0 << endl;
         cerr << "Cannot load image : " << imgLst.filePath( i, lstId ) << endl;
         continue;
      }

      const bool filtered = filterEq( img, val );
      if ( filtered ) nbFilteredSamples++;
      cout << imgLst.filePath( i, lstId ) << 0 << endl;
   }

   cerr << "Filtered : " << nbFilteredSamples << " / " << nbSamples << " ("
        << ( 100.0f * nbFilteredSamples / nbSamples ) << "%)" << endl;

   return ( 0 );
}
