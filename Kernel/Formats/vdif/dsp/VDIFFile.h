//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __VDIFFile_h
#define __VDIFFile_h

#include "dsp/BlockFile.h"

namespace dsp {

  //! Loads BitSeries data from a VDIF file
  /*! Loads data from a file containing raw VLBI Data Interchange Format 
   * (VDIF) packets. */
  class VDIFFile : public BlockFile
  {
  public:
	  
    //! Construct and open file	  
    VDIFFile (const char* filename=0, const char* headername=0);
	  
    //! Destructor
    ~VDIFFile ();
	  
    //! Returns true if file starts with a valid VDIF packet header
    bool is_valid (const char* filename) const;

  protected:

    friend class VDIFUnpacker;

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

#endif // !defined(__VDIFFile_h)
