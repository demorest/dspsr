//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Aidan Hotan
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/spda1k/dsp/spda1k_File.h

#ifndef __SPDA1K_File_h
#define __SPDA1K_File_h

#include "environ.h"

#include "dsp/File.h"

namespace dsp {

  //! Loads TimeSeries data from file.
  class SPDA1K_File : public File 
  {
  public:
    
    //! Construct and open file
    SPDA1K_File (const char* filename = 0);
    
    //! Destructor
    ~SPDA1K_File ();

    //! Return true if filename contains data in the recognized format.
    bool is_valid (const char *filename) const;
    
    
  protected:

    //! Open the file
    void open_file (const char* filename);

    //! Pad over top of gaps in data
    int64_t pad_bytes(unsigned char* buffer, int64_t bytes);

  };
}


#endif // __SPDA1K_File_h
