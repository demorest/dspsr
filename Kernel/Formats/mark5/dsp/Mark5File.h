//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Mark5File_h
#define __Mark5File_h

#include "dsp/BlockFile.h"

namespace dsp {

  //! Loads BitSeries data from a MkV file
  class Mark5File : public BlockFile
  {
  public:
	  
    //! Construct and open file	  
    Mark5File (const char* filename=0, const char* headername=0);
	  
    //! Destructor
    ~Mark5File ();
	  
    //! Returns true if filename is a valid Mk5 file
    bool is_valid (const char* filename, int NOT_USED=-1) const;

  protected:

    friend class Mark5Unpacker;
    friend class Mark5TwoBitCorrection;

    //! Open the file
    void open_file (const char* filename);

    //! Reopen the file
    void reopen ();

    int64 load_bytes (unsigned char* buffer, uint64 nbytes);
    
    int64 seek_bytes (uint64 bytes);

    void* stream;

    uint64 reopen_seek;

  };

}

#endif // !defined(__Mark5File_h)
