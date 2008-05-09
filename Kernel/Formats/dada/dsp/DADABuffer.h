//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __DADABuffer_h
#define __DADABuffer_h

#include "dsp/File.h"
#include "dada_hdu.h"

namespace dsp {

  //! Loads BitSeries data using DADA shared memory
  /*! This class pretends to be a file so that it can slip into the
    File::registry */
  class DADABuffer : public File
  {
    
  public:
    
    //! Constructor
    DADABuffer ();
    
    //! Destructor
    ~DADABuffer ();

    //! Returns true if filename appears to name a valid CPSR file
    bool is_valid (const char* filename,int NOT_USED=-1) const;

    //! Seek to the specified time sample
    virtual void seek (int64 offset, int whence = 0);

    //! Get the information about the data source
    virtual void set_info (const Observation& obs) { info = obs; }

    //! Reset DADAbuffer
    virtual void reset ();
 
  protected:

    //! Read the DADA key information from the specified filename
    virtual void open_file (const char* filename);

    //! Close the DADA connection
    void close ();

    //! Load bytes from shared memory
    virtual int64 load_bytes (unsigned char* buffer, uint64 bytes);
    
    //! Set the offset in shared memory
    virtual int64 seek_bytes (uint64 bytes);

    //! Over-ride File::set_total_samples
    virtual void set_total_samples ();
   
    //! Shared memory interface
    dada_hdu_t* hdu;

    //! Passive viewing mode
    bool passive;

  };

}

#endif // !defined(__DADABuffer_h)
