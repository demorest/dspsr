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

#include "dsp/BlockFile.h"

namespace dsp {

  class PMDAQ_Observation;

  //! Loads BitSeries data from a PMDAQ data file
  class PMDAQFile : public BlockFile 
  {
  public:
   
    //! Construct and open file
    PMDAQFile (const char* filename=0);

    //! Destructor
    virtual ~PMDAQFile();

    //! Returns true if filename appears to name a valid PMDAQ file
    bool is_valid (const char* filename) const;

    //! Which band the bandwidth/centre frequency will be extracted from
    static bool using_second_band;

  protected:

    //! Pads gaps in data
    virtual int64_t pad_bytes(unsigned char* buffer, int64_t bytes);

    //! Open the file
    void open_file (const char* filename);

    //! Initialises 'pmdaq_header' with the header info
    int get_header (char* pmdaq_header, const char* filename); 

    void set_chan_begin(unsigned _chan_begin){ chan_begin = _chan_begin; }
    unsigned get_chan_begin() const { return chan_begin; }
    void set_chan_end(unsigned _chan_end){ chan_end = _chan_end; }
    unsigned get_chan_end() const { return chan_end; }

  private:

    //! If user has use of the second observing band on disk, modify the bandwidth and centre frequency of output
    void modify_info(PMDAQ_Observation* data);

    // Helper functions for load_bytes():

    int chan_begin;
    int chan_end;
  };

}

#endif // !defined(__PMDAQFile_h)
  
