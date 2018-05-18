//*****************************************************************************/
//
// Filename bFaceMModel.h
//
//*****************************************************************************/
#ifndef _FACE_BFACEMMODEL_H
#define _FACE_BFACEMMODEL_H

#include <glm/glm.hpp>
#include <array>
#include <Eigen/Dense>

class BFaceMModel final
{

public :
   BFaceMModel( const std::string& path );

   bool initialized() const {return _valid;}

   enum 
   {
      NumData = 160470,
      NumVertices = 53490,
      NumFaces = 106466,
      NumCoeffs = 199
   };

   // return true if the shape and texture have been generated
   //        false if the model is not valid (provided path is erroneous)
   bool get( const float* shapeCoeffs,
             const float* texCoeffs,
             float* shape,
             float* tex ) const;

   inline const glm::uvec3* getFaces() { return &_faces[0]; }

   static const std::array<size_t,57>& getLandmarksIdx() {return _landmarksIdx;}
             
private:
   bool _valid;

   std::array<glm::uvec3,NumFaces> _faces;

   Eigen::MatrixXf _shapeMU;
   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _shapePC;
   Eigen::Matrix<float,NumCoeffs,1> _shapeEV;

   Eigen::MatrixXf _texMU;
   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _texPC;
   Eigen::Matrix<float,NumCoeffs,1> _texEV;

   static const std::array<size_t,57> _landmarksIdx;
};

#endif // _FACE_BFACEMMODEL_H
