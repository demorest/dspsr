//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/sigproc/dsp/SigProcFile.h,v $
   $Revision: 1.1 $
   $Date: 2008/10/31 05:59:55 $
   $Author: straten $ */


#ifndef __SigProcFile_h
#define __SigProcFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from a SigProc data file
  class SigProcFile : public File 
  {

  public:
   
    //! Construct and open file
    SigProcFile (const char* filename=0);

    //! Returns true if filename appears to name a valid SigProc file
    bool is_valid (const char* filename) const;

    //! Set this to 'false' if you don't need to check bocf
    static bool want_to_check_bocf;

  protected:

    //! Open the file
    virtual void open_file (const char* filename);

    //! Read the SigProc ascii header from filename
    static std::string get_header (const char* filename);

  };

}

#endif // !defined(__SigProcFile_h)
  
