//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/MultiFile.h


#ifndef __MultiFile_h
#define __MultiFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from multiple files
  class MultiFile : public File {

    friend class Multiplex;

  public:
  
    //! Constructor
    MultiFile (const char* name = "MultiFile");
    
    //! Destructor
    virtual ~MultiFile ();

    //! The origin is the current loader
    const Input* get_origin () const { return get_loader(); }

    //! Returns true if filename is an ASCII file listing valid filenames
    bool is_valid (const char* filename) const;
    
    //! Open a number of files and treat them as one logical observation.
    virtual void open (const std::vector<std::string>& new_filenames);

    //! Treat the files as contiguous
    void force_contiguity ();

    //! Makes sure only these filenames are open
    virtual void have_open (const std::vector<std::string>& filenames);

    //! Retrieve a pointer to the loader File instance
    File* get_loader ();
    const File* get_loader () const;

    //! Access to current file objects
    std::vector< Reference::To<File> >& get_files () {return files;}

    //! Return true if the loader File instance is set
    bool has_loader ();

    //! Inquire the number of files
    unsigned nfiles(){ return files.size(); }

    //! Erase the entire list of loadable files
    //! Resets the file pointers
    virtual void erase_files();

    //! Erase just some of the list of loadable files
    //! Resets the file pointers regardless
    virtual void erase_files (const std::vector<std::string>& erase_filenames);

    //! Find out which file is currently open;
    std::string get_current_filename() const { return current_filename; }

    //! Find out the index of current file is
    unsigned get_index() const { return current_index; }

    //! Inquire the next sample to load for the current file
    uint64_t get_next_sample();

    //! Add any relevant extensions (calls loader's add_extensions())
    void add_extensions (Extensions *ext);

  protected:
    
    //! Open the ASCII file of filenames
    virtual void open_file (const char* filename);

    //! Load bytes from file
    virtual int64_t load_bytes (unsigned char* buffer, uint64_t bytes);
    
    //! Adjust the file pointer
    virtual int64_t seek_bytes (uint64_t bytes);

    //! List of files
    std::vector< Reference::To<File> > files;

    //! Currently open File instance
    Reference::To<File> loader;

    //! Name of the currently opened file
    std::string current_filename;

    //! initialize variables
    void init();

    //! Ensure that files are contiguous
    void ensure_contiguity ();

  private:

    //! Test for contiguity
    bool test_contiguity;

    //! Index of the current File in use
    unsigned current_index;

    //! Setup loader and ndat etc after a change to the list of files
    void setup ();

    //! Set the loader to the specified File
    void set_loader (unsigned index);

  };

}

#endif // !defined(__MultiFile_h)
  
