//-*-C++-*-

#ifndef __PMDAQFile_h
#define __PMDAQFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads TimeSeries data from a PMDAQ data file
  class PMDAQFile : public File 
  {
  public:
   
    //! Construct and open file
    PMDAQFile (const char* filename=0);

    //! Returns true if filename appears to name a valid PMDAQ file
    bool is_valid (const char* filename) const;

  protected:
    //! Open the file
    void open_file (const char* filename);
    
    // Insert PMDAQ-specific entries here.
    // Overload these because of fortran 4-byte headers and footers to blocks

    int64 load_bytes (unsigned char * buffer, uint64 bytes);
    int64 seek_bytes (uint64 bytes);

    static int get_header (char* pmdaq_header, const char* filename);

  private:
    int64 absolute_position;

  };

}

#endif // !defined(__PMDAQFile_h)
  
