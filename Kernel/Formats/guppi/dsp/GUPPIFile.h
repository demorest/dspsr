//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __GUPPIFile_h
#define __GUPPIFile_h

#include "dsp/BlockFile.h"

namespace dsp {

  //! Loads BitSeries data from a raw GUPPI file
  class GUPPIFile : public BlockFile
  {
  public:
	  
    //! Construct and open file	  
    GUPPIFile (const char* filename=0);
	  
    //! Destructor
    ~GUPPIFile ();
	  
    //! Returns true if filename is a valid GUPPI file
    bool is_valid (const char* filename) const;

  protected:

    //! Open the file
    void open_file (const char* filename);

    //! Load some data
    int64_t load_bytes (unsigned char *buffer, uint64_t nbytes);

    //! Skip a header
    void skip_extra ();

    //! Temp buffer space for transpose
    unsigned char *tmpbuf;

    //! Header string
    char *hdr;

    //! Number of keys in header string
    int hdr_keys;

  };

}

#endif // !defined(__GUPPIFile_h)
