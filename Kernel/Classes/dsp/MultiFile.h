//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/MultiFile.h,v $
   $Revision: 1.4 $
   $Date: 2002/11/08 01:26:15 $
   $Author: hknight $ */


#ifndef __MultiFile_h
#define __MultiFile_h

#include <vector>

#include "dsp/BitseriesSeekable.h"

namespace dsp {

  class File;

  //! Loads Bitseries data from multiple files
  class MultiFile : public BitseriesSeekable
  {
  public:
    
    //! Constructor
    MultiFile ();
    
    //! Destructor
    virtual ~MultiFile () { }
    
    //! Load a number of files and treat them as one logical observation
    void load (vector<string>& filenames);

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
  
