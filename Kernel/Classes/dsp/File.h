//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/File.h


#ifndef __File_h
#define __File_h

#include "dsp/Seekable.h"
#include "Registry.h"

namespace dsp {

  //! Loads BitSeries data from file
  /*! This class is used in conjunction with the Unpacker class in
    order to add new file formats to the baseband/dsp library.
    Inherit either dsp::File or one of its derived classes and
    implement the two pure virtual methods:

    <UL>
    <LI> bool is_valid()
    <LI> void open_file(const char* filename)
    </UL>

    then register the new class in File_registry.C
  */
  class File : public Seekable
  {
    friend class MultiFile;
    friend class Multiplex;
    friend class HoleyFile;
    friend class RingBuffer;
    
  public:
    
    //! Return a pointer to a new instance of the appropriate sub-class
    /*! This is the entry point for creating new instances of File objects
      from input data files of an arbitrary format. */
    static File* create (const char* filename);

    //! Convenience interface to File::create (const char*)
    static File* create (const std::string& filename)
    { return create (filename.c_str()); }

    //! Constructor
    File (const char* name);
    
    //! Destructor
    virtual ~File ();

    //! Return true if filename contains data in the recognized format
    /*! Derived classes must define the conditions under which they can
      be used to parse the given data file */
    virtual bool is_valid (const char* filename) const = 0;

    //! Open the file
    virtual void open (const char* filename);

    //! Convenience interface to File::open (const char*)
    void open (const std::string& filename) 
    { open (filename.c_str()); }

    //! Close the file
    virtual void close ();

    //! Reopen the file
    virtual void reopen ();

    //! Return the name of the file from which this instance was created
    std::string get_filename () const { return current_filename; }
    std::string get_current_filename() const { return current_filename; }

    //! Inquire how many bytes are in the header
    int get_header_bytes() const{ return header_bytes; }

    //! typedef used to simplify template syntax in File_registry.C
    typedef Registry::List<File> Register;

  protected:
    
    //! Open the file specified by filename for reading
    /*! Derived classes must open the file for reading and set the File::fd,
      File::header_bytes, Input::info, and Input::resolution attributes. */
    virtual void open_file (const char* filename) = 0;  

    //! Return ndat given the file and header sizes, nchan, npol, and ndim
    /*! Called by open_file for some file types, to determine that the
    header ndat matches the file size.  Requires 'info' parameters
    nchan, npol, and ndim as well as header_bytes to be correctly set */
    virtual int64_t fstat_file_ndat(uint64_t tailer_bytes=0);

    //! Over-ride this function to pad data via HoleyFile
    virtual int64_t pad_bytes(unsigned char* buffer, int64_t bytes);

    /** @name Derived Class Defined
     *  These attributes must be set by the open_file method of the
     *  derived class.  */
    //@{
    
    //! The file descriptor
    int fd;
    
    //! The size of the header in bytes
    int header_bytes;

    //@}

    //! The name of the currently opened file, set by open()
    std::string current_filename;

#if HAVE_CUDA
    //! staging buffer for Host to Device transfers
    void * host_buffer;

    //! The size of the host_buffer in bytes
    uint64_t host_buffer_size;
#endif

    //! Load nbyte bytes of sampled data from the device into buffer
    /*! If the data stored on the device contains information other
      than the sampled data, this method should be overloaded and the
      additional information should be filtered out. */
    virtual int64_t load_bytes (unsigned char* buffer, uint64_t nbytes);

#if HAVE_CUDA
    virtual int64_t load_bytes_device (unsigned char* buffer, uint64_t bytes, void * device_handle);
#endif
    
    //! Set the file pointer to the absolute number of sampled data bytes
    /*! If the header_bytes attribute is set, this number of bytes
      will be subtracted by File::seek_bytes before seeking.  If the
      data stored on the device after the header contains information
      other than the sampled data, this method should be overloaded
      and the additional information should be skipped. */
    virtual int64_t seek_bytes (uint64_t bytes);
    
    //! Calculates the total number of samples in the file, based on its size
    virtual void set_total_samples ();

    //! Utility opens the file descriptor
    virtual void open_fd (const std::string& filename);

    //! Return the list of registered sub-classes
    static Register& get_register();

  private:

    //! Initialize variables to sensible null values
    void init();


  };

}

#endif // !defined(__File_h)
  

