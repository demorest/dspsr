//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/File.h,v $
   $Revision: 1.13 $
   $Date: 2003/03/13 23:39:15 $
   $Author: pulsar $ */


#ifndef __File_h
#define __File_h

namespace dsp {
  class File;
}

#include "Registry.h"

#include "dsp/Seekable.h"
#include "dsp/PseudoFile.h"

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

    //! Return a pointer to a new null instance of the derived class 
    virtual File* clone(bool identical=true)=0;

    //! Return true if filename appears to refer to a valid format
    virtual bool is_valid (const char* filename) const = 0;

    //! Open the file
    //! If _info is not null, we copy that data instead of re-reading in all info
    void open (const char* filename, PseudoFile* _info=NULL);

    //! Convenience interface to File::open (const char*)
    void open (const string& filename, PseudoFile* _info=NULL) { open (filename.c_str(),_info); }

    //! Close the file
    virtual void close ();

    //! Return the name of the file from which this instance was created
    string get_filename () const { return current_filename; }
    string get_current_filename(){ return current_filename; }

    //! Return a pointer to a new instance of the appropriate sub-class
    static File* create (const char* filename);

    //! Convenience interface to File::create (const char*)
    static File* create (const string& filename)
    { return create (filename.c_str()); }

    //! Inquire the howmany bytes are in the header
    int get_header_bytes() const{ return header_bytes; }

    //! Return a PseudoFile constructed from this File
    PseudoFile get_pseudofile();

  protected:
    
    //! Called by the wrapper-function, open
    virtual void open_file (const char* filename,PseudoFile* _info) = 0;
    
    //! Called by open_file() for some file types, to determine that the header ndat matches the file size
    //! Requires 'info' parameters nchan,npol,ndim to be set, and also header_bytes to be correctly set
    //! Return value is the actual ndat in the file
    virtual int64 fstat_file_ndat();

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
  

