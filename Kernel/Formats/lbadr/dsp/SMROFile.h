//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Craig West
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/lbadr/dsp/SMROFile.h

#ifndef __SMROFile_h
#define __SMROFile_h

#include "environ.h"

#include "dsp/File.h"

namespace dsp {

  //! Loads TimeSeries data from a SMRO data file.
  class SMROFile : public File 
  {
  public:
    
    //! Construct and open file
    SMROFile (const char* filename = 0);
    
    //! Destructor
    ~SMROFile ();

    //! Return true if filename contains data in the recognized format.
    bool is_valid (const char *filename) const;
    
    
  protected:

    //! Open the file
    void open_file (const char* filename);

    //! Pad over top of gaps in data
    int64_t pad_bytes(unsigned char* buffer, int64_t bytes);

    //! Legacy header format compatibility mode
    bool legacy;
    
  private:
    
    //! Holds the header information for the data file.
    char header[4096]; 

  };
}


#endif // __SMROFile_h
