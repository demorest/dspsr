//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/s2/dsp/S2File.h,v $
   $Revision: 1.5 $
   $Date: 2002/11/09 15:55:27 $
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
   
    //! Returns true if filename appears to name a valid S2 file
    bool is_valid (const char* filename) const;

    //! Construct and open file
    S2File (const char* filename = 0) { if (filename) open (filename); }

  protected:
    //! Open the file
    void open_it (const char* filename);

    // set the number of bytes in header attribute- NOT called by open_it(), but it is called by dsp::ManyFile::switch_to_file()
    virtual void set_header_bytes();
  };

}

#endif // !defined(__S2File_h)
  
