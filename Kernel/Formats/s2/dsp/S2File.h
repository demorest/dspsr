//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/s2/dsp/S2File.h,v $
   $Revision: 1.7 $
   $Date: 2002/11/12 00:24:07 $
   $Author: wvanstra $ */


#ifndef __S2File_h
#define __S2File_h

#include "dsp/File.h"

namespace dsp {

  //! Loads TimeSeries data from a S2 data file.
  /*! The treatment of S2 data is specific to the use of S2-TCI at Swinburne */
  class S2File : public File 
  {
  public:
   
    //! Construct and open file
    S2File (const char* filename = 0);

    //! Returns true if filename appears to name a valid S2 file
    bool is_valid (const char* filename) const;

  protected:
    //! Open the file
    void open_file (const char* filename);

  };

}

#endif // !defined(__S2File_h)
  
