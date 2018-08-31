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

/**
 * @brief a class to handle a file of img filename
 * -> used to access video sequences represented as a list of images
 */
class ImgFileLst
{
  public:
   inline ImgFileLst() { ; }
   inline ImgFileLst( const char* lstFName, const bool check = true )
   {
      (void)open( lstFName, check );
   }
   inline ~ImgFileLst(){};

   /// open a new image file list
   inline bool open( const char* lstFName, const bool check )
   {
      std::ifstream ifs( lstFName );
      if ( ifs.is_open() )
      {
         std::list<std::string> lines;
         while ( ifs.good() )
         {
            std::string line;
            getline( ifs, line );
            if ( check )
            {
               std::ifstream testfs( line.c_str() );
               if ( testfs.is_open() ) lines.push_back( line );
            }
            else
               lines.push_back( line );
         }
         _imgFileNames.resize( lines.size() );
         std::copy( lines.begin(), lines.end(), _imgFileNames.begin() );
         return size() > 0;
      }
      else
      {
         return false;
      }
   }

   inline size_t size() const { return _imgFileNames.size(); }
   inline const std::string& get( size_t i ) const { return _imgFileNames[i]; }

  private:
   std::vector<std::string> _imgFileNames;
};

template <unsigned N = 1>
class ImgNFileLst
{
  public:
   using Data = std::array<std::string, N>;

   enum Opts
   {
      OptsNone = 0,
      OptsCheck = 1 << 1,
      OptsDefault = OptsCheck
   };

   inline ImgNFileLst() { ; }
   inline ImgNFileLst(
       const char* lstFName,
       const char* dataRoot,
       const unsigned flags = OptsDefault )
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
            split( splitLine, line, boost::is_any_of( "\t " ) );
            if ( splitLine.size() >= N )
            {
               Data d;
               bool success = true;
               for ( unsigned s = 0; s < N; ++s )
               {
                  const filesystem::path f( rootPath / filesystem::path( splitLine[s] ) );
                  if ( !doCheck || filesystem::is_regular_file( f ) )
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
         _data.shrink_to_fit();
      }

      return size();
   }

   inline void close() { _data.clear(); }

   inline size_t size() const { return _data.size(); }

   inline const Data& operator[]( size_t i ) const { return _data[i]; }
   inline Data& operator[]( size_t i ) { return _data[i]; }

  private:
   std::vector<Data> _data;
};

class ImgTripletsFileLst
{
  public:
   struct Data
   {
      const float _alpha;
      const std::string _pathA;
      const std::string _pathB;
      const std::string _pathC;

      inline Data(
          const std::string& fA,
          const std::string& fB,
          const std::string& fC,
          const float alpha = 0.5 )
          : _alpha( alpha ), _pathA( fA ), _pathB( fB ), _pathC( fC )
      {
      }
   };

   enum Opts
   {
      OptsNone = 0,
      OptsCheck = 1 << 1,
      OptsAlpha = 1 << 2,
      OptsDefault = OptsCheck
   };

   inline ImgTripletsFileLst() { ; }
   inline ImgTripletsFileLst(
       const char* lstFName,
       const char* dataRoot,
       const unsigned flags = OptsDefault )
   {
      (void)open( lstFName, dataRoot, flags );
   }
   inline ~ImgTripletsFileLst(){};

   /// open a new image file list
   inline size_t
   open( const char* lstFName, const char* dataRoot, const unsigned flags = OptsDefault )
   {
      using namespace std;
      using namespace boost;

      close();

      const bool doCheck = flags & OptsCheck;
      const bool doAlpha = flags & OptsAlpha;

      const unsigned validNbFields = doAlpha ? 4 : 3;

      const filesystem::path rootPath( dataRoot );
      ifstream ifs( lstFName );
      if ( ifs.is_open() )
      {
         _data.reserve( 100000 );
         vector<string> splitLine;
         splitLine.reserve( validNbFields );
         string line;
         while ( ifs.good() )
         {
            getline( ifs, line );
            splitLine.clear();
            split( splitLine, line, boost::is_any_of( "\t " ) );
            if ( splitLine.size() >= validNbFields )
            {
               try
               {
                  // first path :
                  const filesystem::path fA( rootPath / filesystem::path( splitLine[0] ) );
                  const filesystem::path fB( rootPath / filesystem::path( splitLine[1] ) );
                  const filesystem::path fC( rootPath / filesystem::path( splitLine[2] ) );
                  const float alpha( doAlpha ? stof( splitLine[3] ) : 0.5 );
                  if ( !doCheck ||
                       ( filesystem::is_regular_file( fA ) && filesystem::is_regular_file( fB ) &&
                         filesystem::is_regular_file( fC ) ) )
                  {
                     _data.emplace_back( fA.string(), fB.string(), fC.string(), alpha );
                  }
               }
               catch ( ... )
               {
               }
            }
         }
         _data.shrink_to_fit();
      }

      return size();
   }

   inline void close() { _data.clear(); }

   inline size_t size() const { return _data.size(); }

   inline const Data& operator[]( size_t i ) const { return _data[i]; }
   inline Data& operator[]( size_t i ) { return _data[i]; }

  private:
   std::vector<Data> _data;
};

template <0u>
class ImgNFileLst
{
  public:
   using Data = std::vector<std::string>;
   const unsigned N;

   enum Opts
   {
      OptsNone = 0,
      OptsCheck = 1 << 1,
      OptsDefault = OptsCheck
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
            split( splitLine, line, boost::is_any_of( "\t " ) );
            if ( splitLine.size() >= N )
            {
               Data d(N);
               bool success = true;
               for ( unsigned s = 0; s < N; ++s )
               {
                  const filesystem::path f( rootPath / filesystem::path( splitLine[s] ) );
                  if ( !doCheck || filesystem::is_regular_file( f ) )
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
         _data.shrink_to_fit();
      }

      return size();
   }

   inline void close() { _data.clear(); }

   inline size_t size() const { return _data.size(); }

   inline const Data& operator[]( size_t i ) const { return _data[i]; }
   inline Data& operator[]( size_t i ) { return _data[i]; }

  private:
   std::vector<Data> _data;
};

#endif  // _LIBUTILS_IMGFILELST_H
