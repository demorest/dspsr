//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/dsp/CPSRFile.h,v $
   $Revision: 1.3 $
   $Date: 2002/10/15 13:12:31 $
   $Author: pulsar $ */


#ifndef __CPSRFile_h
#define __CPSRFile_h

#include "File.h"

namespace dsp {

  //! Loads Timeseries data from a CPSR data file
  class CPSRFile : public File 
  {
  public:

    //! Returns true if filename appears to name a valid CPSR file
    bool is_valid (const char* filename) const;

    //! Construct and open file
    CPSRFile (const char* filename = 0);

    //! Open the file
    void open (const char* filename);
    
    //! The tape number
    int tapenum;

    //! The file number on tape
    int filenum;

  };

}

#endif // !defined(__CPSRFile_h)
  
