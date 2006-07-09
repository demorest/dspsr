//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/MultiFile.h,v $
   $Revision: 1.23 $
   $Date: 2006/07/09 13:27:10 $
   $Author: wvanstra $ */


#ifndef __MultiFile_h
#define __MultiFile_h

#include "dsp/Seekable.h"

namespace dsp {

  class File;

  //! Loads BitSeries data from multiple files
  class MultiFile : public Seekable {

  public:
  
    //! Constructor
    MultiFile ();
    
    //! Destructor
    virtual ~MultiFile ();
    
    //! Open a number of files and treat them as one logical observation.
    virtual void open (const vector<string>& new_filenames, int bs_index = 0);

    //! Makes sure only these filenames are open
    virtual void have_open (const vector<string>& filenames, int bs_index = 0);

    //! Retrieve a pointer to the loader File instance
    File* get_loader ();

    //! Return true if the loader File instance is set
    bool has_loader ();

    //! Inquire the number of files
    unsigned nfiles(){ return files.size(); }

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
    vector< Reference::To<File> > files;

    //! Currently open File instance
    Reference::To<File> loader;

    //! Name of the currently opened file
    string current_filename;

    //! initialize variables
    void init();

    //! Ensure that files are contiguous
    void ensure_contiguity ();

  private:

    //! Index of the current File in use
    unsigned current_index;

    //! Setup loader and ndat etc after a change to the list of files
    void setup ();

    //! Set the loader to the specified File
    void set_loader (unsigned index);

  };

}

#endif // !defined(__MultiFile_h)
  
