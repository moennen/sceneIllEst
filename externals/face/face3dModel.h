/*! *****************************************************************************
*   \file face3dModel.h
*   \author 2018
*   \brief 
*   \date 2018-04-06
*   *****************************************************************************/
#ifndef _FACE_FACE3DMODEL_H
#define _FACE_FACE3DMODEL_H


class Face3dModel final
{

public :
   Face3dModel( const char* faceModel );
   virtual ~Face3dModel();


   


   bool getMeshFromParams( const double* faceParams,
                           std::vector<float> vtx,
                           std::vector<float> uvs,
                           std::vector<size_t> tridx ) const;




private:
   struct MM3DModel;
   std::unique_ptr<MM3DModel> _modelPtr;
};

#endif // _FACE_FACE3DMODEL_H
