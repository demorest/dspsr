//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/File.h,v $
   $Revision: 1.4 $
   $Date: 2002/10/15 13:12:31 $
   $Author: pulsar $ */


#ifndef __File_h
#define __File_h

#include "Seekable.h"
#include "Registry.h"

namespace dsp {

  //! Loads Timeseries data from file
  class File : public Seekable
  {
  public:
    
    //! Constructor
    File () { init(); }
    
    //! Destructor
    virtual ~File () { }

    //! Return true if filename appears to refer to a valid format
    virtual bool is_valid (const char* filename) const = 0;

    //! Open the file
    virtual void open (const char* filename) = 0;

    //! Convenience interface to File::open (const char*)
    void open (const string& filename) { open (filename.c_str()); }

    //! Close the file
    virtual void close ();

    //! Return a pointer to a new instance of the appropriate sub-class
    static File* create (const char* filename);

    //! Convenience interface to File::create (const char*)
    static File* create (const string& filename)
    { return create (filename.c_str()); }

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

    //! List of registered sub-classes
    static Registry::List<File> registry;

    // Declare friends with Registry entries
    friend class Registry::Entry<File>;

  };

}

#endif // !defined(__File_h)
  

