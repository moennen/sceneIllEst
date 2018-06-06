//*****************************************************************************/
//
// Filename beFaceMModel.h
//
//*****************************************************************************/
#ifndef _FACE_BFACEMMODEL_H
#define _FACE_BFACEMMODEL_H

#include <glm/glm.hpp>
#include <array>
#include <Eigen/Dense>

class BEFaceMModel final
{

public :
   BEFaceMModel( const std::string& path );

   bool initialized() const {return _valid;}

   enum 
   {
      NumData = 159645,
      NumVertices = 53215,
      
      NumFaces = 105840,
      NumCoeffs = 199,
      NumExpCoeffs = 29,
      NumLandmarks = 68
   };

   // return true if the shape and texture have been generated
   //        false if the model is not valid (provided path is erroneous)
   bool get( const float* shapeCoeffs,
             const float* expCoeffs,
             const float* texCoeffs,
             float* shape,
             float* tex,
             const bool normalizedCoeffs = false  ) const;

   inline const glm::uvec3* getFaces() { return &_faces[0]; }

   const std::array<unsigned int,NumLandmarks>& getLandmarksIdx() {return _landmarksIdx;}
             
private:
   bool _valid;

   std::array<glm::uvec3,NumFaces> _faces;

   Eigen::MatrixXf _shapeMU;
   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _shapePC;
   Eigen::Matrix<float,NumCoeffs,1> _shapeEV;

   Eigen::MatrixXf _expMU;
   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _expPC;
   Eigen::Matrix<float,NumExpCoeffs,1> _expEV;

   Eigen::MatrixXf _texMU;
   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _texPC;
   Eigen::Matrix<float,NumCoeffs,1> _texEV;

   std::array<unsigned int,NumLandmarks> _landmarksIdx;
};

#endif // _FACE_BFACEMMODEL_H
