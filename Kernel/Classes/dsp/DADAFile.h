//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/DADAFile.h,v $
   $Revision: 1.2 $
   $Date: 2008/05/28 21:12:42 $
   $Author: straten $ */

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
    static std::string get_header (const char* filename);

  };

}

#endif // !defined(__DADAFile_h)
  
