//*****************************************************************************/
//
// Filename beFaceMModelInstance.C
//*****************************************************************************/

#include <face/beFaceModelInstance.h>

#include <glm/gtx/transform.hpp>

BEFaceMModelInstance::BEFaceMModelInstance( const std::string& path )
    : _valid( false )
{
   using namespace glm;

   FILE* inFile = fopen( path.c_str(), "rb" );
   if ( inFile )
   {
      bool success =
          fread( &_shape[0], sizeof( float ), NumShapeCoeffs, inFile ) == NumShapeCoeffs;
      if ( success ) fread( &_exp[0], sizeof( float ), NumExpCoeffs, inFile ) == NumExpCoeffs;
      if ( success )
         success = fread( &_tex[0], sizeof( float ), NumTexCoeffs, inFile ) == NumTexCoeffs;
      if ( success )
      {
         std::array<float,7> pose;
         success = fread( &pose[0], sizeof( float ), 7, inFile ) == 7 ;

         _pose = glm::rotate(_pose, pose[0], vec3(1.0,0.0,0.0));
         _pose = glm::rotate(_pose, pose[1], vec3(0.0,0.1,0.0));
         _pose = glm::rotate(_pose, pose[2], vec3(0.0,0.0,1.0));
         _pose = glm::scale(_pose, vec3(pose[6], pose[6], pose[6]));
         _pose = glm::translate(_pose, vec3(pose[3], pose[4], pose[5]));
      }
      if ( success ) fread( &_colour[0], sizeof( float ), NumColourParams, inFile ) == NumColourParams;
      if ( success )
         success = fread( &_illum[0], sizeof( float ), NumIllumParams, inFile ) == NumIllumParams;
      
      _valid = success;
      fclose( inFile );
   }
}
