//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __DummyFile_h
#define __DummyFile_h

#include "dsp/BlockFile.h"

namespace dsp {

  //! Make fake data for benchmark purposes
  class DummyFile : public File
  {
  public:
	  
    //! Construct and open file	  
    DummyFile (const char* filename=0, const char* headername=0);
	  
    //! Destructor
    ~DummyFile ();
	  
    //! Returns true if filename is a valid Mk5 file
    bool is_valid (const char* filename) const;

  protected:

    //! Open the file
    void open_file (const char* filename);

    //! close
    void close();

    //! load bytes
    int64_t load_bytes(unsigned char *buffer, uint64_t bytes);

    //! seek bytes
    int64_t seek_bytes(uint64_t bytes);

    void set_total_samples();

#if GENERATE_BIT_PATTERN
  /*
    WvS 2010 05 17 - not sure, but perhaps the data array was intended to
    hold a generated random pattern of bits?
  */
    //! Block of junk data to use
    unsigned char *data;

    //! Current size of data block
    size_t data_size;

    //! Total number of bytes generated
    uint64_t total_bytes;
#endif

    //! Number of bytes to stop after
    uint64_t max_bytes;

  };

}

#endif // !defined(__DummyFile_h)
