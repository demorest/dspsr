//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/sigproc/dsp/SigProcOutputFile.h,v $
   $Revision: 1.2 $
   $Date: 2011/09/19 01:56:42 $
   $Author: straten $ */


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
  
