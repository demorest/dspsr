//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __GUPPIFile_h
#define __GUPPIFile_h

#include "dsp/GUPPIBlockFile.h"

namespace dsp {

  //! Loads BitSeries data from a raw GUPPI file
  class GUPPIFile : public GUPPIBlockFile
  {
  public:
	  
    //! Construct and open file	  
    GUPPIFile (const char* filename=0);
	  
    //! Destructor
    ~GUPPIFile ();
	  
    //! Returns true if filename is a valid GUPPI file
    bool is_valid (const char* filename) const;

    //! Close the file, free memory
    void close ();

  protected:

    //! Open the file
    void open_file (const char* filename);

    //! Load the next hdr/data block
    int load_next_block ();

    //! Seek to a spot in the file
    int64_t seek_bytes (uint64_t bytes);

    //! The correct starting position in the file
    uint64_t pos0;

  };

}

#endif // !defined(__GUPPIFile_h)
