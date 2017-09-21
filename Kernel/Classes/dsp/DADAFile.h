//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/DADAFile.h

#ifndef __DADAFile_h
#define __DADAFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from a DADA data file
  class DADAFile : public File 
  {

  public:
   
    //! Construct and open file
    DADAFile (const char* filename=0);

    //! Returns true if filename appears to name a valid DADA file
    bool is_valid (const char* filename) const;

  protected:

    //! Open the file
    virtual void open_file (const char* filename);

    //! Read the DADA ascii header from filename
    std::string get_header (const char* filename) const;

    //! Flag set true when the header information is in a separate text file
    mutable bool separate_header_file;
  };

}

#endif // !defined(__DADAFile_h)
  
