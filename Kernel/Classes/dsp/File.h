//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/File.h,v $
   $Revision: 1.16 $
   $Date: 2003/07/01 05:27:36 $
   $Author: hknight $ */


#ifndef __File_h
#define __File_h

#include "Registry.h"
#include "dsp/Seekable.h"

namespace dsp {

  class PseudoFile;

  //! Loads BitSeries data from file
  class File : public Seekable
  {
    friend class MultiFile;
    friend class HoleyFile;
    
  public:
    
    //! Return a pointer to a new instance of the appropriate sub-class
    /*! This is the entry point for creating new instances of File objects
      from input data files of an arbitrary format. */
    static File* create (const char* filename);

    //! Convenience interface to File::create (const char*)
    static File* create (const string& filename)
    { return create (filename.c_str()); }

    //! Constructor
    File (const char* name);
    
    //! Destructor
    virtual ~File ();

    //! Return true if filename appears to refer to a file in a valid format
    virtual bool is_valid (const char* filename) const = 0;

    //! Open the file
    void open (const char* filename);

    //! Convenience interface to File::open (const char*)
    void open (const string& filename) 
    { open (filename.c_str()); }

    //! Close the file
    virtual void close ();

    //! Return the name of the file from which this instance was created
    string get_filename () const { return current_filename; }
    string get_current_filename(){ return current_filename; }

    //! Inquire the howmany bytes are in the header
    int get_header_bytes() const{ return header_bytes; }

    //! Return a PseudoFile constructed from this File
    PseudoFile get_pseudofile();

    //! Open from a PseudoFile
    /*! Resets attributes without calling open_file */
    void open (const PseudoFile& file);

  protected:
    
    //! Called by the wrapper-function, open
    virtual void open_file (const char* filename) = 0;  

    //! Return ndat given the file and header sizes, nchan, npol, and ndim
    /*! Called by open_file for some file types, to determine that the
    header ndat matches the file size.  Requires 'info' parameters
    nchan, npol, and ndim as well as header_bytes to be correctly set */
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
    
    //! Calculates the total number of samples in the file, based on its size
    virtual void set_total_samples ();

    //! List of registered sub-classes
    static Registry::List<File> registry;

    // Declare friends with Registry entries
    friend class Registry::Entry<File>;

  private:

    //! Worker function for both forms of open
    void open (const char* filename, const PseudoFile* file);

    //! Initialize variables to sensible null values
    void init();


  };

}

#endif // !defined(__File_h)
  

