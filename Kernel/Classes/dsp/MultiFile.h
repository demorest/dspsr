//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/MultiFile.h,v $
   $Revision: 1.7 $
   $Date: 2002/12/04 01:21:59 $
   $Author: hknight $ */


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
    
    //! Load a number of files and treat them as one logical observation
    void load (vector<string>& filenames);

    void kludge_total_samples(uint64 s){ info.set_ndat(s); }

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
  
