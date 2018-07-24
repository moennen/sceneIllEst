//*****************************************************************************/
//
// Filename beFaceMModel.C
//*****************************************************************************/

#include <face/beFaceMModel.h>

BEFaceMModel::BEFaceMModel( const std::string& path, const bool withUVs )
    : _valid( false ),
      _shapeMU( (int)NumData, 1 ),
      _shapePC( (int)NumData, (int)NumCoeffs ),
      _expMU( (int)NumData, 1 ),
      _expPC( (int)NumData, (int)NumExpCoeffs ),
      _texMU( (int)NumData, 1 ),
      _texPC( (int)NumData, (int)NumCoeffs )
{
   FILE* inFile = fopen( path.c_str(), "rb" );
   if ( inFile )
   {
      bool success =
          fread( &_landmarksIdx[0], sizeof( unsigned int ), NumLandmarks, inFile ) == NumLandmarks;
      if ( success )
         success =
             fread( &_faces[0], sizeof( unsigned int ), NumFaces * 3, inFile ) == NumFaces * 3;
      if ( success )
         success = fread( _shapeMU.data(), sizeof( float ), NumData, inFile ) == NumData;
      if ( success )
         success = fread( _shapePC.data(), sizeof( float ), NumData * NumCoeffs, inFile ) ==
                   NumData * NumCoeffs;
      if ( success )
         success = fread( _shapeEV.data(), sizeof( float ), NumCoeffs, inFile ) == NumCoeffs;
      if ( success ) fread( _texMU.data(), sizeof( float ), NumData, inFile ) == NumData;
      if ( success )
         success = fread( _texPC.data(), sizeof( float ), NumData * NumCoeffs, inFile ) ==
                   NumData * NumCoeffs;
      if ( success )
         success = fread( _texEV.data(), sizeof( float ), NumCoeffs, inFile ) == NumCoeffs;
      if ( success ) success = fread( _expMU.data(), sizeof( float ), NumData, inFile ) == NumData;
      if ( success )
         success = fread( _expPC.data(), sizeof( float ), NumData * NumExpCoeffs, inFile ) ==
                   NumData * NumExpCoeffs;
      if ( success )
         success = fread( _expEV.data(), sizeof( float ), NumExpCoeffs, inFile ) == NumExpCoeffs;
      if ( success )
         success = !withUVs || fread( &_uvs[0], sizeof( float ), NumVertices * 2, inFile ) == NumVertices * 2;

      _valid = success;
      fclose( inFile );
   }
}

BEFaceMModel::BEFaceMModel( const std::string& path, const glm::vec2* uvs ) :
  BEFaceMModel(path, false)
{
  if ( _valid ) std::memcpy(&_uvs[0], uvs, sizeof(float)*NumVertices*2 );
}

bool BEFaceMModel::save( const std::string& path )
{
   if ( !_valid ) return false;
   FILE* outFile = fopen( path.c_str(), "wb" );
   if ( !outFile ) return false;
   bool success =
       fwrite( &_landmarksIdx[0], sizeof( unsigned int ), NumLandmarks, outFile ) == NumLandmarks;
   if ( success )
      success = fwrite( &_faces[0], sizeof( unsigned int ), NumFaces * 3, outFile ) == NumFaces * 3;
   if ( success ) success = fwrite( _shapeMU.data(), sizeof( float ), NumData, outFile ) == NumData;
   if ( success )
      success = fwrite( _shapePC.data(), sizeof( float ), NumData * NumCoeffs, outFile ) ==
                NumData * NumCoeffs;
   if ( success )
      success = fwrite( _shapeEV.data(), sizeof( float ), NumCoeffs, outFile ) == NumCoeffs;
   if ( success ) fwrite( _texMU.data(), sizeof( float ), NumData, outFile ) == NumData;
   if ( success )
      success = fwrite( _texPC.data(), sizeof( float ), NumData * NumCoeffs, outFile ) ==
                NumData * NumCoeffs;
   if ( success )
      success = fwrite( _texEV.data(), sizeof( float ), NumCoeffs, outFile ) == NumCoeffs;
   if ( success ) success = fwrite( _expMU.data(), sizeof( float ), NumData, outFile ) == NumData;
   if ( success )
      success = fwrite( _expPC.data(), sizeof( float ), NumData * NumExpCoeffs, outFile ) ==
                NumData * NumExpCoeffs;
   if ( success )
      success = fwrite( _expEV.data(), sizeof( float ), NumExpCoeffs, outFile ) == NumExpCoeffs;
   if ( success )
      success = fwrite( &_uvs[0], sizeof( float ), NumVertices * 2, outFile ) == NumVertices * 2;
   fclose( outFile );
   return success;
}

bool BEFaceMModel::get(
    const float* shapeCoeffs,
    const float* expCoeffs,
    const float* texCoeffs,
    float* shapeBuff,
    float* texBuff,
    const bool normalizedCoeffs ) const
{
   if ( !_valid ) return false;

   using namespace Eigen;

   const auto sc = Map<Matrix<float, NumCoeffs, 1> >( const_cast<float*>( shapeCoeffs ) );
   const auto ec = Map<Matrix<float, NumExpCoeffs, 1> >( const_cast<float*>( expCoeffs ) );
   const auto tc = Map<Matrix<float, NumCoeffs, 1> >( const_cast<float*>( texCoeffs ) );

   auto shape = Map<Matrix<float, NumData, 1> >( shapeBuff );
   auto tex = Map<Matrix<float, NumData, 1> >( texBuff );

   const float sf = 0.001;

   if ( normalizedCoeffs )
   {
      shape = ( _shapeMU + _expMU + _shapePC * sc + _expPC * ec ) * sf;
      tex = ( _texMU + _texPC * tc ) * sf;
   }
   else
   {
      shape = ( _shapeMU + _expMU + _shapePC * ( _shapeEV.asDiagonal() * sc ) +
                _expPC * ( _expEV.asDiagonal() * ec * 0.01 ) ) *
              sf;
      tex = tex = ( _texMU + _texPC * ( _texEV.asDiagonal() * tc ) ) * sf;
   }

   return true;
}
