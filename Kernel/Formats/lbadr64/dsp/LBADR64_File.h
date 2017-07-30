//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Craig West & Aidan Hotan
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/lbadr64/dsp/LBADR64_File.h

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
    bool is_valid (const char *filename) const;
    
    
  protected:

    //! Open the file
    void open_file (const char* filename);

    //! Pad over top of gaps in data
    int64_t pad_bytes(unsigned char* buffer, int64_t bytes);

  private:
    
    //! Holds the header information for the data file.
    char header[4096]; 

  };
}


#endif // __LBADR64_File_h
