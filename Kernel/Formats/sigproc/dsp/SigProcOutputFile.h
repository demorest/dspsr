//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/sigproc/dsp/SigProcOutputFile.h


#ifndef __SigProcOutputFile_h
#define __SigProcOutputFile_h

#include "dsp/OutputFile.h"

namespace dsp {

  //! Loads BitSeries data from a SigProc data file
  class SigProcOutputFile : public OutputFile 
  {
  public:
   
    //! Construct and open file
    SigProcOutputFile (const char* filename=0);

  protected:

    //! Write the file header to the open file
    void write_header ();

    //! Get the extension to be added to the end of new filenames
    std::string get_extension () const;

  };

}

#endif // !defined(__SigProcOutputFile_h)
  
