//-*-C++-*-

#ifndef __Mark5File_h
#define __Mark5File_h

#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from a MkV file
  class Mark5File : public File
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
		
    int64 load_bytes (unsigned char* buffer, uint64 nbytes);
    
    int64 seek_bytes (uint64 bytes);

    void* stream;

  };

}

#endif // !defined(__Mark5File_h)
