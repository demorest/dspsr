//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2014 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __LWAFile_h
#define __LWAFile_h

#include "dsp/BlockFile.h"

#define LWA_HEADER_BYTES 32
#define LWA_DATA_BYTES 4096
#define LWA_FRAME_BYTES (LWA_HEADER_BYTES+LWA_DATA_BYTES)

namespace dsp {

  //! Loads BitSeries data from a LWA DRX file
  /*! Loads data from a file containing LWA DRX data format 
   * Reference: http://fornax.phys.unm.edu/lwa/trac/wiki/DP_Formats
   */
  class LWAFile : public BlockFile
  {
  public:
	  
    //! Construct and open file	  
    LWAFile (const char* filename=0, const char* headername=0);
	  
    //! Destructor
    ~LWAFile ();
	  
    //! Returns true if file starts with a valid LWA packet header
    bool is_valid (const char* filename) const;

  protected:

    friend class LWAUnpacker;

    //! Open the file
    void open_file (const char* filename);

    //! Reopen the file
    void reopen ();
    
    //int64_t seek_bytes (uint64_t bytes);

    void* stream;

    uint64_t reopen_seek;

    char datafile[256];

  };

}

#endif // !defined(__LWAFile_h)
