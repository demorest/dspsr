//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/MultiFile.h,v $
   $Revision: 1.18 $
   $Date: 2003/09/27 08:36:28 $
   $Author: hknight $ */


#ifndef __MultiFile_h
#define __MultiFile_h

#include <vector>

#include "dsp/Observation.h"
#include "dsp/Seekable.h"
#include "dsp/PseudoFile.h"

namespace dsp {

  //! Loads BitSeries data from multiple files
  class MultiFile : public Seekable {

  public:
  
    //! Constructor
    MultiFile ();
    
    //! Destructor
    virtual ~MultiFile ();
    
    //! Open a number of files and treat them as one logical observation.
    //! 'bs_index' is the index of the BitSeries to be loaded.  Usually there is only a single BitSeries stream per file, but for classes such as MultiBitSeriesFile there can be multiple streams per file
    virtual void open (const vector<string>& new_filenames, int bs_index = -1);

    //! Makes sure only these filenames are open
    //! Resets the file pointers
    virtual void have_open (const vector<string>& filenames, int bs_index = -1);

    //! Use to open files when they've already been opened once
    virtual void open(const vector<PseudoFile*>& pseudos);

    //! Retrieve a pointer to the loader File instance
    File* get_loader(){ if(!loader) return NULL; return loader.get(); }

    //! Retrieve a pointer to the pseudofile
    PseudoFile* get_file(unsigned ifile){ return &files[ifile]; }

    //! Inquire the number of files
    unsigned nfiles(){ return files.size(); }

    PseudoFile* get_first_file(){ return &files[0]; }
    PseudoFile* get_last_file(){ return &files.back(); }

    //! Erase the entire list of loadable files
    //! Resets the file pointers
    virtual void erase_files();

    //! Erase just some of the list of loadable files
    //! Resets the file pointers regardless
    virtual void erase_files (const vector<string>& erase_filenames);

    //! Find out which file is currently open;
    string get_current_filename() const { return current_filename; }

    //! Find out the index of current file is
    unsigned get_index() const { return current_index; }

    //! Inquire the next sample to load for the current file
    uint64 get_next_sample();

  protected:
    
    //! Load bytes from file
    virtual int64 load_bytes (unsigned char* buffer, uint64 bytes);
    
    //! Adjust the file pointer
    virtual int64 seek_bytes (uint64 bytes);

    //! List of files
    vector<PseudoFile> files;

    //! Currently open File instance
    Reference::To<File> loader;

    //! Name of the currently opened file
    string current_filename;

    //! initialize variables
    void init();

    //! Ensure that files are contiguous
    void ensure_contiguity();

  private:

    //! Index of the current PseudoFile in use
    unsigned current_index;

    //! Setup loader and ndat etc after a change to the list of files
    void setup ();

    //! Set the loader to the specified PseudoFile
    void set_loader (unsigned index);

  };

}

#endif // !defined(__MultiFile_h)
  
