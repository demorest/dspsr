//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Craig West
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/mark4/dsp/Mark4File.h

#ifndef __Mark4File_h
#define __Mark4File_h

#include "dsp/File.h"

int decode_bcd(char in[4]);

namespace dsp {

  enum Modes { Standard,
	       VLBA
  };
  
  //! Loads TimeSeries data from a Mark4 data file.
  class Mark4File : public File 
  {
  public:
    
    //! Construct and open file
    Mark4File (const char* filename = 0);
    
    //! Destructor
    ~Mark4File ();

    //! Return true if filename contains data in the recognized format.
    bool is_valid (const char *filename) const;
    
    
  protected:

    //! Open the file
    void open_file (const char* filename);
    
    //! Locates the next SYNC pulse, relative to the 'from' position.
    uint64_t find_sync (int file_descriptor, uint64_t from = 0) const;

    //! Initalises the data stream.
    void initalise ();

    //! Evaluates how many channels of data are present - assumes file open
    int count_channels (int file_descriptor) const;

    //! Returns the MJD date of the next header.
    MJD decode_date(uint64_t from = 0);
    
    //! Defines the number of data channels.
    unsigned int channels;

    //! Defines the SYNC pattern to search for.
    char SYNC_pattern;

    //! Defines the mode the data was recorded in.
    Modes mode;
    
  private:
    
    //! Holds the start time for the data file.
    //! Note: This is TEMPORARY and will contact a full header in the future.
    char timestamp[17]; 
    
    //! Used to determine if data is VLBA or non-VLBA formated.
    bool VLBA_mode;
    
    
  };
}


#endif // __Mark4File_h
