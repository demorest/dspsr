//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/File.h,v $
   $Revision: 1.3 $
   $Date: 2002/10/11 12:12:22 $
   $Author: mbailes $ */


#ifndef __File_h
#define __File_h

#include "Seekable.h"

namespace dsp {

  //! Loads Timeseries data from file
  class File : public Seekable
  {
  public:
    
    //! Constructor
    File () { init(); }
    
    //! Destructor
    virtual ~File () { }

    //! Open the file
    virtual void open (const char* filename) = 0;

    //! Convenience interface
    void open (const string& filename) { open (filename.c_str()); }

    //! Close the file
    virtual void close ();

  protected:
    
    //! The file descriptor
    int fd;
    
    //! Size of the header in bytes
    int header_bytes;
    
    //! Load nbyte bytes of sampled data from the device into buffer
    /*! If the data stored on the device contains information other
      than the sampled data, this method should be overloaded and the
      additional information should be filtered out. */
    virtual int64 load_bytes (unsigned char* buffer, uint64 nbyte);
    
    //! Set the file pointer to the absolute number of sampled data bytes
    /*! If the header_bytes attribute is set, this number of bytes
      will be subtracted by File::seek_bytes before seeking.  If the
      data stored on the device after the header contains information
      other than the sampled data, this method should be overloaded
      and the additional information should be skipped. */
    virtual int64 seek_bytes (uint64 bytes);
    
    //! initialize variables
    void init();
  };

}

#endif // !defined(__File_h)
  

