//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/dsp/CPSRFile.h,v $
   $Revision: 1.8 $
   $Date: 2002/11/12 00:24:07 $
   $Author: wvanstra $ */


#ifndef __CPSRFile_h
#define __CPSRFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads TimeSeries data from a CPSR data file
  class CPSRFile : public File 
  {
  public:

    //! Construct and open file
    CPSRFile (const char* filename = 0);

    //! Returns true if filename appears to name a valid CPSR file
    bool is_valid (const char* filename) const;

    //! The tape number
    int tapenum;

    //! The file number on tape
    int filenum;

  protected:
    //! Open the file
    virtual void open_file (const char* filename);

  };

}

#endif // !defined(__CPSRFile_h)
  
