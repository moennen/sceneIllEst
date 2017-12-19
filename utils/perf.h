//*****************************************************************************/
//
// Filename perf.h
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
#ifndef _LIBUTILS_PERF_H
#define _LIBUTILS_PERF_H

#include <time.h>

/**
 * @brief a class to compute and integrate execution time
 *
 * usage : Perf::start { c1 } Perf::stop 
 *         Perf::start { c2 } Perf::stop
 *         Perf::getMs -> return the number of milliseconds used to 
 *                        compute c1 and c2
 *
 *         Perf::start { c3 } Perf::stop
 *         Perf::getMs -> return the number of milliseconds used to 
 *                        compute c1 and c2 and c3
 *
 *         Perf::reset
 *         Perf::start { c4 } Perf::stop
 *         Perf::getMs -> return the number of milliseconds used to 
 *                        compute c4
 *
 *         Perf::start { c5 } Perf::start { c6 } Perf::stop
 *         Perf::getMs -> return the number of milliseconds used to 
 *                        compute c4 and c6 (and start cancel a previous start)
 */
class Perf {

public :
   Perf(clockid_t clockId=CLOCK_REALTIME) :_clockId(clockId) {reset();};
   ~Perf(){}

   inline void start() 
   {
      _started=true;
      clock_gettime(_clockId,&_startT);
   }

   inline void stop()  
   {
      timespec endT;
      clock_gettime(_clockId,&endT);
      if (_started)
      {
         _nsamples+=1.0;
         _msecs_last= (endT.tv_nsec < _startT.tv_nsec) ?
            1000.0*(endT.tv_sec-_startT.tv_sec-1) + 0.000001*(_startT.tv_nsec-endT.tv_nsec) :
            1000.0*(endT.tv_sec-_startT.tv_sec) + 0.000001*(endT.tv_nsec-_startT.tv_nsec);
         _msecs_sum+=_msecs_last;
         _started=false;
      }
   }

   inline void reset() 
   {
      _started=false;
      _msecs_last=0.0;
      _msecs_sum=0.0;
      _nsamples=0.0;
   }

   inline double getMs(bool cumul) const 
   {
      return (cumul?_msecs_sum:_msecs_sum/_nsamples);
   }

   inline double getMs() const 
   {
      return _msecs_last;
   }

private :
   bool _started;
   timespec _startT;
   double _msecs_last;
   double _msecs_sum;
   double _nsamples;
   clockid_t _clockId;
};

#endif // _LIBUTILS_PERF_H
