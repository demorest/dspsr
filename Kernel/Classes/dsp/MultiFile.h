//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/MultiFile.h,v $
   $Revision: 1.1 $
   $Date: 2002/09/19 07:59:56 $
   $Author: wvanstra $ */


#ifndef __MultiFile_h
#define __MultiFile_h

#include <vector>

#include "Seekable.h"

namespace dsp {

  //! Loads Timeseries data from multiple files
  class MultiFile : public Seekable
  {
  public:
    
    //! Constructor
    MultiFile ();
    
    //! Destructor
    virtual ~MultiFile () { }
    
  protected:
    
    //! Load bytes from file
    virtual int64 load_bytes (unsigned char* buffer, uint64 bytes);
    
    //! Adjust the file pointer
    virtual int64 seek_bytes (uint64 bytes);

    //! Current file as set by seek_bytes
    int index;

    //! Offset number of bytes of data into current file
    uint64 offset;

    //! File descriptors of open data files
    vector<int> fds;

    //! Bytes of data in each file
    vector<uint64> data_bytes;

    //! Bytes in header of each file
    vector<uint64> header_bytes;

    //! Names of data files
    vector<string> fnames;

    //! Return the index of the file containing the offset_from_obs_start byte
    /*! offsets do no include header_bytes */
    int getindex (int64 offset_from_obs_start, int64& offset_in_file);

    //! initialize variables
    void init();
  };

}

#endif // !defined(__MultiFile_h)
  
