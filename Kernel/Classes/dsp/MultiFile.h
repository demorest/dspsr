//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/MultiFile.h,v $
   $Revision: 1.11 $
   $Date: 2003/02/13 01:28:48 $
   $Author: pulsar $ */


#ifndef __MultiFile_h
#define __MultiFile_h

#include <vector>

#include "dsp/Observation.h"
#include "dsp/Seekable.h"
#include "dsp/File.h"
#include "dsp/PseudoFile.h"

namespace dsp {

  class File;

  //! Loads BitSeries data from multiple files
  class MultiFile : public Seekable {

  public:
    
    //! Constructor
    MultiFile ();
    
    //! Destructor
    virtual ~MultiFile () { }
    
    //! Open a number of files and treat them as one logical observation.  This function will take the union of the existing filenames and the new ones, and sort them by start time
    //! Resets the file pointers
    virtual void open (vector<string> new_filenames);

    //! Makes sure only these filenames are open
    //! Resets the file pointers
    virtual void have_open(vector<string> filenames);

    //! Retrieve a pointer to the loader File instance
    File* get_loader(){ if(!loader) return NULL; return loader.get(); }

    //! Retrieve a pointer to the pseudofile
    Observation* get_file(unsigned ifile){ return &files[ifile]; }

    //! Inquire the number of files
    unsigned nfiles(){ return files.size(); }

    //! Erase the entire list of loadable files
    //! Resets the file pointers
    virtual void erase_files();

    //! Erase just some of the list of loadable files
    //! Resets the file pointers regardless
    virtual void erase_files(vector<string> erase_filenames);

    //! Find out which file is currently open;
    string get_current_filename(){ return current_filename; }

    //! Find out the index of current file is
    unsigned get_index(){ return index; }

  protected:
    
    //! Load bytes from file
    virtual int64 load_bytes (unsigned char* buffer, uint64 bytes);
    
    //! Adjust the file pointer
    virtual int64 seek_bytes (uint64 bytes);

    // List of files
    vector<PseudoFile> files;

    //! Loader
    Reference::To<File> loader;
    
    //! Current File in use
    unsigned index;

    //! Return the index of the file containing the offset_from_obs_start byte
    /*! offsets do no include header_bytes */
    int getindex (int64 offset_from_obs_start, int64& offset_in_file);

    //! initialize variables
    void init();

    //! Ensure that files are contiguous
    void ensure_contiguity();

    //! Setup ndat etc after a change to the list of files
    void setup(Reference::To<dsp::File> opener);

    //! The currently open filename (not really needed but we have it anyway)
    string current_filename;

  };

}

#endif // !defined(__MultiFile_h)
  
