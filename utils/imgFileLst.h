//*****************************************************************************/
//
// Filename imgFileLst.h
//
// Copyright (c) 2014 Autodesk, Inc.
// All rights reserved.
// 
// This computer source code and related instructions and comments are the 
// unpublished confidential and proprietary information of Autodesk, Inc.
// and are protected under applicable copyright and trade secret law.
// They may not be disclosed to, copied or used by any third party without 
// the prior written consent of Autodesk, Inc.
//*****************************************************************************/
#ifndef _LIBUTILS_IMGFILELST_H
#define _LIBUTILS_IMGFILELST_H

#include <list>
#include <vector>
#include <string>
#include <fstream>

/**
 * @brief a class to handle a file of img filename
 * -> used to access video sequences represented as a list of images
 */
class ImgFileLst {

public :
   inline ImgFileLst() {;} 
   inline ImgFileLst(const char *lstFName, const bool check=true) 
   {
      (void)open(lstFName,check);
   }
   inline ~ImgFileLst() {};

   /// open a new image file list
   inline bool open(const char *lstFName, const bool check)
   {
      std::ifstream ifs(lstFName);
      if (ifs.is_open())
      {
         std::list< std::string > lines;
         while ( ifs.good() )
         {
            std::string line;
            getline (ifs,line);
            if (check)
            {
               std::ifstream testfs(line.c_str());
               if (testfs.is_open()) lines.push_back(line);
            }
            else lines.push_back(line);
         }
         _imgFileNames.resize(lines.size());
         std::copy(lines.begin(),lines.end(),_imgFileNames.begin());
         return size() > 0;
      }
      else 
      {
         return false;
      }
   }

   inline size_t size() const {return _imgFileNames.size();}
   inline const std::string &get(size_t i) const  {return _imgFileNames[i];}

private :
   std::vector< std::string > _imgFileNames;
};

#endif // _LIBUTILS_IMGFILELST_H
