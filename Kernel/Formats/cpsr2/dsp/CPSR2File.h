//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr2/dsp/CPSR2File.h,v $
   $Revision: 1.1 $
   $Date: 2002/09/19 07:59:56 $
   $Author: wvanstra $ */


#ifndef __CPSR2File_h
#define __CPSR2File_h

#include "File.h"

namespace dsp {

  //! Loads Timeseries data from file
  class CPSR2File : public File 
  {
  public:
    
    //! Open the file
    void open (const char* filename);

  };

}

#endif // !defined(__CPSR2File_h)
  
