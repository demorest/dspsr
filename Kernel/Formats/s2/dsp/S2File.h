//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/s2/dsp/S2File.h,v $
   $Revision: 1.1 $
   $Date: 2002/10/04 10:37:09 $
   $Author: wvanstra $ */


#ifndef __S2File_h
#define __S2File_h

#include "File.h"

namespace dsp {

  //! Loads Timeseries data from a S2 data file.
  /*! The treatment of S2 data is specific to the use of S2-TCI at Swinburne */
  class S2File : public File 
  {
  public:
   
    //! Returns true if filename appears to name a valid S2 file
    static bool is_valid (const char* filename);

    //! Construct and open file
    S2File (const char* filename = 0) { if (filename) open (filename); }

    //! Open the file
    void open (const char* filename);

  };

}

#endif // !defined(__S2File_h)
  
