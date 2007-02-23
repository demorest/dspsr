//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Craig West & Aidan Hotan
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/lbadr64/dsp/LBADR64_File.h,v $
   $Revision: 1.1 $
   $Date: 2007/02/23 04:29:36 $
   $Author: ahotan $ */

#ifndef __LBADR64_File_h
#define __LBADR64_File_h

#include "environ.h"

#include "dsp/File.h"

namespace dsp {

  //! Loads TimeSeries data from a LBADR64 data file.
  class LBADR64_File : public File 
  {
  public:
    
    //! Construct and open file
    LBADR64_File (const char* filename = 0);
    
    //! Destructor
    ~LBADR64_File ();

    //! Return true if filename contains data in the recognized format.
    bool is_valid (const char *filename,int NOT_USED=-1) const;
    
    
  protected:

    //! Open the file
    void open_file (const char* filename);

    //! Pad over top of gaps in data
    int64 pad_bytes(unsigned char* buffer, int64 bytes);

  private:
    
    //! Holds the header information for the data file.
    char header[4096]; 

  };
}


#endif // __LBADR64_File_h
