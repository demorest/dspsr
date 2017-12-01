//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __GUPPIBlockFile_h
#define __GUPPIBlockFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from GUPPI data blocks
  /*! This class implements a general scheme for loading and 
   * parsing data organized in raw guppi_daq blocks.  Derived 
   * classes will be able to either read the data directly
   * from a file, or from a guppi_daq shared memory ring buffer.
   */
  class GUPPIBlockFile : public File
  {
  public:
	  
    //! Construct and open file	  
    GUPPIBlockFile (const char* name);
	  
    //! Destructor
    ~GUPPIBlockFile ();
	  
    //! Returns true if filename is a valid GUPPI file
    virtual bool is_valid (const char* filename) const = 0;

    //! Return true if data are signed
    bool get_signed() const { return signed_8bit; }

  protected:

    //! Open the file
    virtual void open_file (const char* filename) = 0;

    //! Send data bytes to unpacker
    int64_t load_bytes (unsigned char *buffer, uint64_t nbytes);

    //! Load next hdr/data block
    virtual int load_next_block () = 0;

    //! Parse the current header into info struct
    void parse_header ();

    //! Pointer to current header
    char *hdr;

    //! Pointer to current data block
    unsigned char *dat;

    //! Number of keys in header string
    int hdr_keys;

    //! Have the data been transposed already
    bool time_ordered;

    //! Are 8-bit data signed or unsigned
    bool signed_8bit;

    //! Size of current data block in bytes
    uint64_t blocsize;

    //! Overlap between blocks, in samples (per channel)
    uint64_t overlap;

    //! Location in current data block
    uint64_t current_block_byte;

    //! Current block packet index
    int64_t current_pktidx;

    //! Number of bytes of zeros to emit
    int64_t current_nzero_byte;

    //! Packet size
    int packet_size;

  };

}

#endif // !defined(__GUPPIFile_h)
