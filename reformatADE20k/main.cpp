/*!
 * *****************************************************************************
 *   \file main.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-02-19
 *   *****************************************************************************/

#include "utils/imgFileLst.h"
#include "utils/cv_utils.h"

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
extern array<unsigned char, 3148> classIdMap;

//------------------------------------------------------------------------------
//
inline unsigned char remapLabel( const Vec3b& label )
{
   const auto id = ( static_cast<unsigned short>( label.x ) / 10 ) * 256 +
                   static_cast<unsigned short>( label.y );

   return id < classIdMap.size() ? classIdMap[id] : 255;
}
//
void remapLabels( const array<Mat, 3>& in_labels, Mat& out_labels, Mat& out_instances )
{
   const ivec2 sz = {in_labels[0].cols, in_labels[0].rows};
   out_labels = Mat::zeros( sz.y, sz.x, CV_8UC3 );
   out_instances = Mat::zeros( sz.y, sz.x, CV_8UC3 );

#pragma omp parallel for
   for ( unsigned y = 0; y < sz.y; y++ )
   {
      const Vec3b* il_ptr[3] = {
          in_labels[0].ptr<Vec3b>( y ), in_labels[1].ptr<Vec3b>( y ), in_labels[2].ptr<Vec3b>( y )};
      Vec3b* ol_ptr = out_labels.ptr<Vec3b>( y );
      Vec3b* oi_ptr = out_instances.ptr<Vec3b>( y );
      for ( unsigned x = 0; x < sz.x; x++ )
      {
         ol_ptr[x] = {
             remapLabel( il_ptr[0][x] ), remapLabel( il_ptr[1][x] ), remapLabel( il_ptr[2][x] )};
         oi_ptr[x] = {il_ptr[0][x].z, il_ptr[1][x].z, il_ptr[2][x].z};
      }
   }
}

}  // namespace

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@imgFileLst   |         | images list   }"
    "{@imgRootDir   |         | images root dir   }"
    "{@imgOutDir    |         | images output dir   }"
    "{show           |         |    }"
    "{wait           | 10        |    }"
    "{nowrite        |        |    }";

int main( int argc, char* argv[] )
{
   CommandLineParser parser( argc, argv, keys );
   if ( parser.has( "help" ) )
   {
      parser.printMessage();
      return ( 0 );
   }

   const bool doShow = parser.get<bool>( "show" );
   const int wait = parser.get<int>( "wait" );
   const bool doWrite = !parser.get<bool>( "nowrite" );
   const bool toLinear = false;

   const filesystem::path outRootPath( parser.get<string>( "@imgOutDir" ) );
   filesystem::path outGroupRootPath = outRootPath;

   // Create the list of 1 image + 3 labels
   ImgNFileLst<4> imgLst(
       parser.get<string>( "@imgFileLst" ).c_str(), parser.get<string>( "@imgRootDir" ).c_str() );
   if ( imgLst.size() == 0 )
   {
      cerr << "Invalid dataset : " << parser.get<string>( "@imgFileLst" ) << endl;
      return -1;
   }

   unsigned startIdx = 0;
   const int nMaxRendersPerGroup = 10000;

   // Loop through the data
   const size_t nbSamples = imgLst.size();
   for ( size_t i = 0; i < nbSamples; ++i )
   {
      Mat img = imread( imgLst.filePath( i, 0 ), IMREAD_UNCHANGED );
      if ( img.empty() )
      {
         cerr << "Cannot load image : " << imgLst.filePath( i, 0 ) << endl;
         continue;
      }

      // create output directory
      if ( ( ( startIdx + i ) % nMaxRendersPerGroup ) == 0 )
      {
         const unsigned renderGroupId = ( startIdx + i ) / nMaxRendersPerGroup;
         char dirname[7];
         sprintf( dirname, "%06d", renderGroupId );
         outGroupRootPath = outRootPath / filesystem::path( std::string( dirname ) );
         if ( !filesystem::create_directory( outGroupRootPath ) )
         {
            cerr << "Cannot create directory : " << outGroupRootPath.string() << endl;
         }
      }

      char sampleIdName[7];
      sprintf( sampleIdName, "%06d", i );
      const string outPrefix = string( sampleIdName ) + "_";

      // Read input label images (the first one has to exist)
      array<Mat, 3> in_labels;
      for ( size_t j = 0; j < 3; ++j )
      {
         in_labels[j] = imread( imgLst.filePath( i, j + 1 ), IMREAD_UNCHANGED );
         if ( in_labels[j].empty() )
         {
            if ( j == 0 )
            {
               cerr << "Cannot load image : " << imgLst.filePath( i, 1 ) << endl;
               break;
            }
            in_labels[j] = Mat::zeros( in_labels[0].rows, in_labels[0].cols, CV_8UC3 );
         }
      }
      if ( in_labels[2].empty() ) continue;

      Mat out_labels, out_instances;
      remapLabels( in_labels, out_labels, out_instances );

      const string outBasename = outPrefix;
      const filesystem::path fImg( outBasename + string( "_rgb" ) + ".png" );
      const filesystem::path fLabels( outBasename + string( "_label" ) + ".png" );
      const filesystem::path fInstances( outBasename + string( "_inst" ) + ".png" );

      if ( doWrite )
      {
         imwrite( filesystem::path( outGroupRootPath / fImg ).string().c_str(), img );
         imwrite( filesystem::path( outGroupRootPath / fLabels ).string().c_str(), out_labels );
         imwrite(
             filesystem::path( outGroupRootPath / fInstances ).string().c_str(), out_instances );
      }

      cout << fImg.string() << " " << fLabels.string() << " " << fInstances.string() << endl;

      if ( doShow )
      {
         imshow( "InLabels_0", in_labels[0] );
         imshow( "InLabels_1", in_labels[1] );
         imshow( "InLabels_2", in_labels[2] );

         imshow( "OutLabels", out_labels );
         imshow( "OutInstances", out_instances );

         waitKey( wait );
      }
   }

   return ( 0 );
}
