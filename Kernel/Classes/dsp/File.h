//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/File.h,v $
   $Revision: 1.1 $
   $Date: 2002/09/19 07:59:56 $
   $Author: wvanstra $ */


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
    
    //! Load bytes from file
    virtual int64 load_bytes (unsigned char* buffer, uint64 bytes);
    
    //! Set the file pointer
    virtual int64 seek_bytes (uint64 bytes);
    
    //! initialize variables
    void init();
  };

}

#endif // !defined(__File_h)
  
