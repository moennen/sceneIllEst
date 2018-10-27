/*!
 * *****************************************************************************
 *   \file genIntegratedFlowMap.cpp
 *   \author moennen
 *   \brief
 *   \date 2018-10-01
 *   *****************************************************************************/

#include "utils/imgFileLst.h"
#include "utils/cv_utils.h"

#include <libopticalFlow/oclVarOpticalFlow.h>

#include <glm/glm.hpp>

#include <boost/filesystem.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;
using namespace boost;

namespace
{

//------------------------------------------------------------------------------
//
void resizeToMax( Mat& img, const uvec2 sampleSz )
{
   uvec2 imgSz( img.cols, img.rows );
   // random rescale
   const float ds = std::max( (float)sampleSz.y / imgSz.y, (float)sampleSz.x / imgSz.x );
   resize( img, img, Size(), ds, ds, CV_INTER_AREA );
}

//------------------------------------------------------------------------------
//
Mat flowToWrite( const Mat& flow, const Mat&flowErr )
{
  vector<Mat> splitFlow(3); split(flow, &splitFlow[0]);
  splitFlow[2] = flowErr;
  Mat out(flow.rows, flow.cols, CV_32FC3);
  merge(&splitFlow[0],3,out);
  return out;
}

//------------------------------------------------------------------------------
//
Mat flowToImg( const Mat& flow, const bool leg = false )
{
   Mat flow_split[2];
   Mat t_split[3];
   Mat bgr;
   split( flow, flow_split );
   if ( leg )
   {
      t_split[0] = Mat::zeros( flow_split[0].size(), flow_split[0].type() );
      t_split[1] = flow_split[1];
      t_split[2] = flow_split[0];
      merge( t_split, 3, bgr );
   }
   else
   {
      Mat hsv;
      Mat magnitude, angle;
      cartToPolar( flow_split[0], flow_split[1], magnitude, angle, true );
      normalize( magnitude, magnitude, 0, 1, NORM_MINMAX );
      t_split[0] = angle;  // already in degrees - no normalization needed
      t_split[1] = Mat::ones( angle.size(), angle.type() );
      t_split[2] = magnitude;
      merge( t_split, 3, hsv );
      cvtColor( hsv, bgr, COLOR_HSV2BGR );
   }
   return bgr;
}

//------------------------------------------------------------------------------
//
Mat flowToErr( const Mat& flow, const Mat& imgFrom, const Mat& imgTo )
{
   Mat lk( flow.rows, flow.cols, CV_32FC1 );

#pragma omp parallel for
   for ( size_t y = 0; y < flow.rows; y++ )
   {
      float* lk_data = lk.ptr<float>( y );
      const vec2* flow_data = flow.ptr<vec2>( y );
      const vec3* from_data = imgFrom.ptr<vec3>( y );

      for ( size_t x = 0; x < flow.cols; x++ )
      {
         const vec2 f = flow_data[x];
         const vec3 from = from_data[x];
         const vec3 to = cv_utils::imsample32FC3<vec3>( imgTo, vec2( x, y ) + f );
         const vec3 diff = ( from - to ); 
         lk_data[x] = dot(diff,diff);
      }
   }

   return lk;
}

//------------------------------------------------------------------------------
//
Mat integrateFlow( const Mat& flow0To1, const Mat& flow1ToN )
{
   Mat flow0ToN(flow0To1.rows, flow0To1.cols, CV_32FC2);

#pragma omp parallel for
   for ( size_t y = 0; y < flow0To1.rows; y++ )
   {
      const vec2* flow0To1_data = flow0To1.ptr<vec2>( y );
      vec2* flow0ToN_data = flow0ToN.ptr<vec2>( y );
      for ( size_t x = 0; x < flow0To1.cols; x++ )
      {
         const vec2 uv0To1 = flow0To1_data[x];
         const vec2 uv1ToN = cv_utils::imsample32F<vec2>( flow1ToN, vec2(x,y) + uv0To1 );
         flow0ToN_data[x] = uv0To1 + uv1ToN;
      }
   }

   return flow0ToN;
}

}

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@imgFileLst    |         | images list   }"
    "{@imgRootDir    |         | images root dir   }"
    "{@imgOutDir     |         | images output dir   }"
    "{seqLength      |7        | number of images to integrated per sequence}"
    "{outWidth       |640      | output width }"
    "{outHeight      |380      | output height }"
    "{maxFlowMSErr   |0.5      | maximum tolerance for the flow mean square error}"
    "{backward       |         |    }"
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

   const unsigned uSeqLenght = parser.get<unsigned>( "seqLength" );
   const uvec2 minSz =
       uvec2( parser.get<unsigned>( "outWidth" ), parser.get<unsigned>( "outHeight" ) );

   const bool toLinear = false;
   const bool doBackward = parser.get<bool>( "backward" );
   const bool doShow = parser.get<bool>( "show" );
   const bool doWrite = !parser.get<bool>( "nowrite" );

   const float fMaxFlowMSErr = parser.get<float>( "maxFlowMSErr" );

   const filesystem::path outRootPath( parser.get<string>( "@imgOutDir" ) );

   // Create the list of image triplets
   ImgNFileLst imgLst(
       uSeqLenght,
       parser.get<string>( "@imgFileLst" ).c_str(),
       parser.get<string>( "@imgRootDir" ).c_str() );
   if ( imgLst.size() == 0 )
   {
      cerr << "Invalid dataset : " << parser.get<string>( "@imgFileLst" ) << endl;
      return -1;
   }

   // Create the optical flow estimator
   OclVarOpticalFlow::params_t ofParams = OclVarOpticalFlow::getDefaultParams();
   ofParams.nonLinearIter = 9;
   ofParams.robustIter = 5;
   ofParams.solverIter = 5;
   ofParams.lambda = 0.17;
   //ofParams.gamma = 150;

   OclVarOpticalFlow ofEstimator( minSz.x, minSz.y, false, ofParams );

   for ( size_t i = 0; i < imgLst.size(); ++i )
   {
      const string outBasename = filesystem::path( imgLst( i, doBackward ? uSeqLenght-1 : 0 ) ).stem().string();
    
      Mat matRefImg = cv_utils::imread32FC3( imgLst.filePath( i, doBackward ? uSeqLenght-1 : 0 ), toLinear, true );
      if (matRefImg.empty())
      {
        cerr << "Cannot load image : " << imgLst.filePath( i, doBackward ? uSeqLenght-1 : 0 ) << endl;
        continue; 
      }
      
      Mat matCurrImg = matRefImg.clone();
      GaussianBlur( matCurrImg, matCurrImg, Size( 3, 3 ), 0.113 );
      Mat matFlowCurrToRef = Mat(matCurrImg.rows, matCurrImg.cols, CV_32FC2, 0.0f);
      Mat matFlowCurrToPrev( matFlowCurrToRef.rows, matFlowCurrToRef.cols, CV_32FC2 );

      ofEstimator.setImgSize( matFlowCurrToRef.cols, matFlowCurrToRef.rows );

      bool success = true;
      for ( size_t j = 1; j < uSeqLenght; ++j )
      {
        Mat matPrevImg = matCurrImg;
        Mat matCurrImg = cv_utils::imread32FC3( imgLst.filePath( i, doBackward ? uSeqLenght-(j+1) : j ), toLinear, true );  
        if (matCurrImg.empty())
        {
          success = false;
          cerr << "Cannot load image : " << imgLst.filePath( i, doBackward ? uSeqLenght-(j+1) : j ) << endl;
          break;
        }
        GaussianBlur( matCurrImg, matCurrImg, Size( 3, 3 ), 0.113 );
  
        ofEstimator.compute(
          reinterpret_cast<const float*>( matPrevImg.ptr() ),
          reinterpret_cast<const float*>( matCurrImg.ptr() ),
          matFlowCurrToPrev.cols,
          matFlowCurrToPrev.rows,
          reinterpret_cast<float*>( matFlowCurrToPrev.ptr() ) );

        matFlowCurrToRef = integrateFlow(matFlowCurrToPrev, matFlowCurrToRef);
      }

      if (!success) continue;

      // reread the current image to get it clean (with no preprocessing)
      matCurrImg = cv_utils::imread32FC3( imgLst.filePath( i, doBackward ? 0 : uSeqLenght-1 ), toLinear, true  );  

      // check the error 
      Mat matFlowErr = flowToErr( matFlowCurrToRef, matCurrImg, matRefImg);
      if (cv::mean(matFlowErr)[0] > fMaxFlowMSErr ) continue;

      cerr << "MSERR : " << cv::mean(matFlowErr)[0] << endl;

      const filesystem::path fImgRef( outBasename + "_ref.png" );
      const filesystem::path fImgCurr( outBasename + "_curr.png" );
      const filesystem::path fFlow( outBasename + string( "_flow" ) + ".exr" );
      
      cvtColor(matRefImg, matRefImg, COLOR_BGR2RGB );
      cvtColor(matCurrImg, matCurrImg, COLOR_BGR2RGB );
      
      if ( doWrite )
      {
         imwrite( filesystem::path( outRootPath / fImgRef ).string().c_str(), matRefImg * 255.0 );
         imwrite( filesystem::path( outRootPath / fImgCurr ).string().c_str(), matCurrImg * 255.0 );
         imwrite( filesystem::path( outRootPath / fFlow ).string().c_str(), flowToWrite(matFlowCurrToRef,matFlowErr) );
      }

      cout << fImgRef.string() << " " << fImgCurr.string() << " " << fFlow.string() << " " << uSeqLenght << endl;
           
      // display
      if ( doShow )
      {
         imshow( "Reference", matRefImg );
         imshow( "Current", matCurrImg );
         imshow( "Flow", flowToImg(matFlowCurrToRef) );
         waitKey( cv::mean(matFlowErr)[0] > 0.15 ?  0 : 10 );
      }
   }

   return ( 0 );
}
