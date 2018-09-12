/*! *****************************************************************************
 *   \file imgFileLst.h
 *   \author moennen
 *   \brief
 *   \date 2014-05-05
 *   *****************************************************************************/
#ifndef _LIBUTILS_IMGFILELST_H
#define _LIBUTILS_IMGFILELST_H

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <list>
#include <vector>
#include <string>
#include <fstream>

class ImgNFileLst
{
  public:
   using Data = std::vector<std::string>;
   const unsigned N;

   enum Opts
   {
      OptsNone = 0,
      OptsCheck = 1 << 1,
      OptsDefault = OptsNone
   };

   inline ImgNFileLst( const unsigned n ) : N( n ) { ; }
   inline ImgNFileLst(
       const unsigned n,
       const char* lstFName,
       const char* dataRoot,
       const unsigned flags = OptsDefault )
       : N( n )
   {
      (void)open( lstFName, dataRoot, flags );
   }
   inline ~ImgNFileLst(){};

   /// open a new image file list
   inline size_t
   open( const char* lstFName, const char* dataRoot, const unsigned flags = OptsDefault )
   {
      using namespace std;
      using namespace boost;

      close();

      const bool doCheck = flags & OptsCheck;

      const filesystem::path rootPath( dataRoot );
      ifstream ifs( lstFName );
      if ( ifs.is_open() )
      {
         _data.reserve( 100000 );
         vector<string> splitLine;
         splitLine.reserve( N );
         string line;
         while ( ifs.good() )
         {
            getline( ifs, line );
            splitLine.clear();
            split( splitLine, line, is_any_of( "\t " ) );
            if ( splitLine.size() >= N )
            {
               Data d( N );
               bool success = true;
               for ( unsigned s = 0; s < N; ++s )
               {
                  const filesystem::path f( splitLine[s] );
                  if ( !doCheck || filesystem::is_regular_file( rootPath / f ) )
                  {
                     d[s] = f.string();
                  }
                  else
                  {
                     success = false;
                     break;
                  }
               }
               if ( success ) _data.push_back( d );
            }
         }
         _rootPath = rootPath;
         _data.shrink_to_fit();
      }

      return size();
   }

   inline void close() { _data.clear(); }

   inline size_t size() const { return _data.size(); }

   inline const Data& operator[]( size_t i ) const { return _data[i]; }
   inline const std::string& operator()( size_t i, size_t j ) const { return _data[i][j]; }

   inline std::string filePath( size_t i, size_t j ) const
   {
      return ( _rootPath / boost::filesystem::path( _data[i][j] ) ).string();
   }
   inline const boost::filesystem::path& rootPath() const { return _rootPath; }

  private:
   boost::filesystem::path _rootPath;
   std::vector<Data> _data;
};

#endif  // _LIBUTILS_IMGFILELST_H
