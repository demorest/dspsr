//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/File.h,v $
   $Revision: 1.11 $
   $Date: 2002/11/12 00:23:29 $
   $Author: wvanstra $ */


#ifndef __File_h
#define __File_h

#include "dsp/Seekable.h"
#include "Registry.h"

namespace dsp {

  //! Loads BitSeries data from file
  class File : public Seekable
  {
    friend class MultiFile;

  public:
    
    //! Constructor
    File (const char* name);
    
    //! Destructor
    virtual ~File ();

    //! Return true if filename appears to refer to a valid format
    virtual bool is_valid (const char* filename) const = 0;

    //! Open the file
    void open (const char* filename);

    //! Convenience interface to File::open (const char*)
    void open (const string& filename) { open (filename.c_str()); }

    //! Close the file
    virtual void close ();

    //! Return the name of the file from which this instance was created
    string get_filename () const { return current_filename; }

    //! Return a pointer to a new instance of the appropriate sub-class
    static File* create (const char* filename);

    //! Convenience interface to File::create (const char*)
    static File* create (const string& filename)
    { return create (filename.c_str()); }

    string get_current_filename(){ return current_filename; }

  protected:
    
    //! Called by the wrapper-function, open
    virtual void open_file (const char* filename) = 0;

    //! The file descriptor
    int fd;
    
    //! Size of the header in bytes
    int header_bytes;
    
    //! The name of the currently opened file, set by open()
    string current_filename;

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
    
    //! Calculates the total number of samples in the file, base on it size
    virtual void set_total_samples ();

    //! initialize variables to sensible null values
    void init();

    //! List of registered sub-classes
    static Registry::List<File> registry;

    // Declare friends with Registry entries
    friend class Registry::Entry<File>;

  };

}

#endif // !defined(__File_h)
  

