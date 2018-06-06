//*****************************************************************************/
//
// Filename beFaceMModelInstance.h
//
//*****************************************************************************/
#ifndef _FACE_BEFACEMMODEL_INSTANCE_H
#define _FACE_BEFACEMMODEL_INSTANCE_H

#include <glm/glm.hpp>
#include <array>
#include <Eigen/Dense>

class BEFaceMModelInstance final
{

public :

   enum 
   {
      NumShapeCoeffs = 199,
      NumExpCoeffs = 29,
      NumTexCoeffs = 199,
      NumColourParams = 7,
      NumIllumParams = 10
   };

   BEFaceMModelInstance( const std::string& path );

   bool initialized() const {return _valid;}

   inline const float* getShapeCoeffs() const {return  &_shape[0]; }
   inline const float* getExpCoeffs() const { return &_exp[0]; }
   inline const float* getTexCoeffs() const { return &_tex[0]; }

   inline glm::mat4 getPose() const { return _pose; }
             
private:
   bool _valid;


   glm::mat4 _pose;

   std::array<float, NumShapeCoeffs> _shape;
   std::array<float, NumExpCoeffs> _exp;
   std::array<float, NumTexCoeffs> _tex;

   std::array<float, NumColourParams> _colour;
   std::array<float, NumIllumParams> _illum;
};

#endif // _FACE_BFACEMMODEL_H
