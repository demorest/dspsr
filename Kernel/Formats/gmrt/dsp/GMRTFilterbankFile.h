//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/gmrt/dsp/GMRTFilterbankFile.h


#ifndef __GMRTFilterbankFile_h
#define __GMRTFilterbankFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from a GMRT filterbank data file.
  class GMRTFilterbankFile : public File 
  {
  public:
   
    //! Construct and open file
    GMRTFilterbankFile (const char* filename = 0);

    //! Returns true if filename appears to name a valid GMRTFilterbank file
    bool is_valid (const char* filename) const;

  protected:

    //! Open the file
    void open_file (const char* filename);

  };

}

#endif // !defined(__GMRTFilterbankFile_h)
  
