//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr2/dsp/CPSR2File.h,v $
   $Revision: 1.7 $
   $Date: 2002/11/10 12:57:08 $
   $Author: wvanstra $ */


#ifndef __CPSR2File_h
#define __CPSR2File_h

#include "dsp/File.h"

namespace dsp {

  //! Loads TimeSeries data from a CPSR2 data file
  class CPSR2File : public File 
  {
  public:
   
    //! Construct and open file
    CPSR2File (const char* filename=0);

    //! Returns true if filename appears to name a valid CPSR2 file
    bool is_valid (const char* filename) const;
    
    static int get_header (char* cpsr2_header, const char* filename);

  protected:

    //! Open the file
    virtual void open_it (const char* filename);

    // set the number of bytes in header attribute- called by open_it() and by dsp::ManyFile::switch_to_file()
    virtual void set_header_bytes();

  };

}

#endif // !defined(__CPSR2File_h)
  
