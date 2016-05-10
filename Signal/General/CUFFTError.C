/***************************************************************************
 *
 *   Copyright (C) 2014 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "CUFFTError.h"

#include <stdio.h>
#include <stdarg.h>
#include <iostream>
#include <cstdio>

using namespace std;

const char* cufftResult_to_string (cufftResult result)
{
  cerr << "cufftResult_to_string result=" << result << endl;

  switch (result)
    {
    case CUFFT_SUCCESS :
      return "The CUFFT operation was successful";
    case CUFFT_INVALID_PLAN :
      return "CUFFT was passed an invalid plan handle";
    case CUFFT_ALLOC_FAILED :
      return "CUFFT failed to allocate GPU or CPU memory";
    case CUFFT_INVALID_TYPE :
      return "No longer used";
    case CUFFT_INVALID_VALUE :
      return "User specified an invalid pointer or parameter";
    case CUFFT_INTERNAL_ERROR :
      return "Driver or internal CUFFT library error";
    case CUFFT_EXEC_FAILED :
      return "Failed to execute an FFT on the GPU";
    case CUFFT_SETUP_FAILED :
      return "The CUFFT library failed to initialize";
    case CUFFT_INVALID_SIZE :
      return "User specified an invalid transform size";
    case CUFFT_UNALIGNED_DATA :
      return "No longer used";
#if CUDA_VERSION >= 5050
    case CUFFT_INCOMPLETE_PARAMETER_LIST :
      return "Missing parameters in call";
    case CUFFT_INVALID_DEVICE :
      return "Execution of a plan was on different GPU than plan creation";
    case CUFFT_PARSE_ERROR :
      return "Internal plan database error";
    case CUFFT_NO_WORKSPACE :
      return "No workspace has been provided prior to plan execution";
#endif
#if CUDA_VERSION >= 6050
    case CUFFT_NOT_IMPLEMENTED:
      return "Not Implemented";
    case CUFFT_LICENSE_ERROR:
      return "License error";
#endif
    }
  return "unrecognized cufftResult";
}

CUFFTError::CUFFTError (cufftResult r, const char* func, const char* msg, ...)
{
  char buf[1024];
  string this_msg;

  if (msg) {
    va_list args;
  
    va_start(args, msg);
    vsnprintf(buf, 1024, msg, args);
    va_end(args);
    this_msg = buf;
  }

  this_msg += ": ";
  this_msg += cufftResult_to_string (r);

  construct (FailedCall, func, this_msg.c_str());
} 

CUFFTError::CUFFTError (cufftResult r, const char* func, const string& msg)
{
  string this_msg = msg;
  this_msg += ": ";
  this_msg += cufftResult_to_string (r);

  construct (FailedCall, func, this_msg.c_str());
}
