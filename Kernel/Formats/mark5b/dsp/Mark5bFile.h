//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2016 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Mark5bFile_h
#define __Mark5bFile_h

#include "dsp/BlockFile.h"

namespace dsp {

  //! Loads BitSeries data from a MkV file using the mark5access library
  class Mark5bFile : public BlockFile
  {
  public:
	  
    //! Construct and open file	  
    Mark5bFile (const char* filename=0, const char* headername=0);
	  
    //! Destructor
    ~Mark5bFile ();
	  
    //! Returns true if filename is a valid Mk5 file
    bool is_valid (const char* filename) const;

  protected:

    friend class Mark5bUnpacker;
    friend class Mark5bTwoBitCorrection;

    //! Open the file
    void open_file (const char* filename);

    //! Reopen the file
    void reopen ();

    int64_t load_bytes (unsigned char* buffer, uint64_t nbytes);
    
    int64_t seek_bytes (uint64_t bytes);

    void* stream;

    uint64_t reopen_seek;

  };

}

#endif // !defined(__Mark5bFile_h)
