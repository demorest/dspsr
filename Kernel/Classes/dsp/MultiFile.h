//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/MultiFile.h,v $
   $Revision: 1.9 $
   $Date: 2003/01/13 11:44:03 $
   $Author: pulsar $ */


#ifndef __MultiFile_h
#define __MultiFile_h

#include <vector>

#include "dsp/Seekable.h"
#include "dsp/File.h"

namespace dsp {

  class File;

  //! Loads BitSeries data from multiple files
  class MultiFile : public Seekable
  {
  public:
    
    //! Constructor
    MultiFile ();
    
    //! Destructor
    virtual ~MultiFile () { }
    
    //! Open a number of files and treat them as one logical observation
    void open (vector<string>& filenames);

    //! Retrieve a pointer to one of the File instances
    File* get_file(unsigned ifile){ return files[ifile]; }

    //! Inquire the number of files
    unsigned nfiles(){ return files.size(); }

  protected:
    
    //! Load bytes from file
    virtual int64 load_bytes (unsigned char* buffer, uint64 bytes);
    
    //! Adjust the file pointer
    virtual int64 seek_bytes (uint64 bytes);

    //! File instances
    vector< Reference::To<File> > files;

    //! Current File in use
    unsigned index;

    //! Return the index of the file containing the offset_from_obs_start byte
    /*! offsets do no include header_bytes */
    int getindex (int64 offset_from_obs_start, int64& offset_in_file);

    //! initialize variables
    void init();
  };

}

#endif // !defined(__MultiFile_h)
  
