//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Matthew Bailes
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __PMDAQFile_h
#define __PMDAQFile_h

#include "environ.h"

#include "dsp/File.h"
#include "dsp/Observation.h"
#include "dsp/PMDAQ_Observation.h"

namespace dsp {

  //! Loads BitSeries data from a PMDAQ data file
  class PMDAQFile : public File 
  {
  public:
   
    //! Construct and open file
    PMDAQFile (const char* filename=0);

    //! Returns true if filename appears to name a valid PMDAQ file
    bool is_valid (const char* filename,int NOT_USED=-1) const;

    //! Which band the bandwidth/centre frequency will be extracted from
    static bool using_second_band;

  protected:

    //! Pads gaps in data
    virtual int64 pad_bytes(unsigned char* buffer, int64 bytes);

    //! Open the file
    void open_file (const char* filename);

    // Insert PMDAQ-specific entries here.
    // Overload these because of fortran 4-byte headers and footers to blocks

    //! Loads 'bytes' bytes of data into 'buffer'
    virtual int64 load_bytes (unsigned char * buffer, uint64 bytes);

    //! Seeks over 'bytes' bytes of data
    virtual int64 seek_bytes (uint64 bytes);

    //! Initialises 'pmdaq_header' with the header info
    int get_header (char* pmdaq_header, const char* filename); 

    void set_chan_begin(unsigned _chan_begin){ chan_begin = _chan_begin; }
    unsigned get_chan_begin() const { return chan_begin; }
    void set_chan_end(unsigned _chan_end){ chan_end = _chan_end; }
    unsigned get_chan_end() const { return chan_end; }

  private:

    //! If user has use of the second observing band on disk, modify the bandwidth and centre frequency of output
    void modify_info(PMDAQ_Observation* data);

    // Sets the ndat of the file from the filesize
    void work_out_ndat(const char* filename);

    // Helper functions for load_bytes():

    //! Returns how many bytes can be loaded before hitting the end of the file 
    uint64 bytes_available();

    //! Loads in the first partial chunk
    uint64 load_partial_chunk(unsigned char*& buffer, uint64 bytes);
    //! Loads in a full chunk
    uint64 load_chunk(unsigned char*& buffer);
    //! Loads in the last partial chunk
    uint64 load_last_chunk(unsigned char*& buffer, uint64 bytes);

    //! Seeks over header/trailer
    void seek_ahead();

    //! Sets the end_of_data flag
    int64 cleanup(uint64 bytes_loaded);

    //! Should be the same number as return value from lseek(fd,0,SEEK_CUR)- ie number of bytes from start of file
    int64 absolute_position;

    uint64 header_bytes;
    uint64 data_bytes;
    uint64 trailer_bytes;
    //#define header_bytes 4
    //#define data_bytes (48*1024)
    //#define trailer_bytes 4

    int chan_begin;
    int chan_end;
  };

}

#endif // !defined(__PMDAQFile_h)
  
