//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/dsp/CPSRFile.h,v $
   $Revision: 1.6 $
   $Date: 2002/11/09 15:55:25 $
   $Author: wvanstra $ */


#ifndef __CPSRFile_h
#define __CPSRFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads TimeSeries data from a CPSR data file
  class CPSRFile : public File 
  {
  public:

    //! Returns true if filename appears to name a valid CPSR file
    bool is_valid (const char* filename) const;

    //! Construct and open file
    CPSRFile (const char* filename = 0);

    //! The tape number
    int tapenum;

    //! The file number on tape
    int filenum;

  protected:
    //! Open the file
    virtual void open_it (const char* filename);

    // set the number of bytes in header attribute- called by open_it() and by dsp::ManyFile::switch_to_file()
    virtual void set_header_bytes();
    
  };

}

#endif // !defined(__CPSRFile_h)
  
