//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr2/dsp/CPSR2File.h,v $
   $Revision: 1.16 $
   $Date: 2006/07/09 13:27:07 $
   $Author: wvanstra $ */


#ifndef __CPSR2File_h
#define __CPSR2File_h

#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from a CPSR2 data file
  class CPSR2File : public File 
  {
  public:
   
    //! Construct and open file
    CPSR2File (const char* filename=0);

    //! Returns true if filename appears to name a valid CPSR2 file
    bool is_valid (const char* filename,int NOT_USED=-1) const;

    //! Set this to 'false' if you don't need to yamasaki verify
    static bool want_to_yamasaki_verify;

  protected:

    //! Pads gaps in data
    virtual int64 pad_bytes(unsigned char* buffer, int64 bytes);
      
    //! Open the file
    virtual void open_file (const char* filename);

    //! Read the CPSR2 ascii header from filename
    static int get_header (char* cpsr2_header, const char* filename);
  };

}

#endif // !defined(__CPSR2File_h)
  
