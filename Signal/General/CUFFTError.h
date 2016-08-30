//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2014 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __CUFFTError_h
#define __CUFFTError_h

#include "Error.h"
#include <cufft.h>

class CUFFTError : public Error {

  public:

  //! Error with optional printf-style message
  CUFFTError (cufftResult, const char* func, const char* msg=0, ...);

  //! Error with string message
  CUFFTError (cufftResult, const char* func, const std::string& msg);

  //! Destructor
  ~CUFFTError () {}

};

#endif
