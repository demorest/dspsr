//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/dsp/CPSRFile.h,v $
   $Revision: 1.1 $
   $Date: 2002/09/19 07:59:56 $
   $Author: wvanstra $ */


#ifndef __CPSRFile_h
#define __CPSRFile_h

#include "File.h"

namespace dsp {

  //! Loads Timeseries data from file
  class CPSRFile : public File 
  {
  public:
    
    //! Open the file
    void open (const char* filename);
    
    //! The tape number
    int tapenum;

    //! The file number on tape
    int filenum;

  };

}

#endif // !defined(__CPSRFile_h)
  
