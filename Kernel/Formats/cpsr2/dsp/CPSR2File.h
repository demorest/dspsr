//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr2/dsp/CPSR2File.h,v $
   $Revision: 1.4 $
   $Date: 2002/11/03 21:51:49 $
   $Author: wvanstra $ */


#ifndef __CPSR2File_h
#define __CPSR2File_h

#include "dsp/File.h"

namespace dsp {

  //! Loads Timeseries data from a CPSR2 data file
  class CPSR2File : public File 
  {
  public:
   
    //! Returns true if filename appears to name a valid CPSR2 file
    bool is_valid (const char* filename) const;
    
    static int get_header (char* cpsr2_header, const char* filename);

    //! Construct and open file
    CPSR2File (const char* filename=0) { if (filename) open (filename); }

    //! Open the file
    void open (const char* filename);

  };

}

#endif // !defined(__CPSR2File_h)
  
