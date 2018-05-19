//*****************************************************************************/
//
// Filename bFaceMModel.C
//*****************************************************************************/

#include <face/bFaceMModel.h>

const std::array<size_t, 57> BFaceMModel::_landmarksIdx = {
    8309,  8319,  6611,  6244, 5879,  7032,  8334,  4410,  4404,  4416,  4402,  4418,  2088,  5959,
    8344,  7570,  5768,  5006, 8374,  6697,  21628, 19963, 19330, 20203, 20547, 40514, 40087, 38792,
    9965,  10372, 10781, 9612, 12150, 12144, 12156, 12142, 12158, 10603, 11714, 33496, 35151, 35807,
    34981, 34656, 8354,  8366, 40777, 14472, 9118,  10928, 10051, 41091, 41511, 42825};

BFaceMModel::BFaceMModel( const std::string& path )
    : _valid( false ),
      _shapeMU( (int)NumData, 1 ),
      _shapePC( (int)NumData, (int)NumCoeffs ),
      _texMU( (int)NumData, 1 ),
      _texPC( (int)NumData, (int)NumCoeffs )
{
   FILE* inFile = fopen( path.c_str(), "rb" );
   if ( inFile )
   {
      bool success =
          fread( &_faces[0], sizeof( unsigned int ), NumFaces * 3, inFile ) == NumFaces * 3;
      if ( success ) fread( _shapeMU.data(), sizeof( float ), NumData, inFile ) == NumData;
      if ( success )
         success = fread( _shapeEV.data(), sizeof( float ), NumCoeffs, inFile ) == NumCoeffs;
      if ( success )
         success = fread( _shapePC.data(), sizeof( float ), NumData * NumCoeffs, inFile ) ==
                   NumData * NumCoeffs;
      if ( success ) fread( _texMU.data(), sizeof( float ), NumData, inFile ) == NumData;
      if ( success )
         success = fread( _texEV.data(), sizeof( float ), NumCoeffs, inFile ) == NumCoeffs;
      if ( success )
         success = fread( _texPC.data(), sizeof( float ), NumData * NumCoeffs, inFile ) ==
                   NumData * NumCoeffs;

      _valid = success;
      fclose( inFile );
   }
}

bool BFaceMModel::get(
    const float* shapeCoeffs,
    const float* texCoeffs,
    float* shapeBuff,
    float* texBuff ) const
{
   if ( !_valid ) return false;

   using namespace Eigen;

   const auto sc = Map<Matrix<float, NumCoeffs, 1> >( const_cast<float*>( shapeCoeffs ) );
   const auto tc = Map<Matrix<float, NumCoeffs, 1> >( const_cast<float*>( texCoeffs ) );

   auto shape = Map<Matrix<float, NumData, 1> >( shapeBuff );
   auto tex = Map<Matrix<float, NumData, 1> >( texBuff );

   shape = ( _shapeMU + _shapePC * ( _shapeEV.asDiagonal() * sc ) ) / 1000.0;
   tex = ( _texMU + _texPC * ( _texEV.asDiagonal() * tc ) ) / 1000.0;

   return true;
}
