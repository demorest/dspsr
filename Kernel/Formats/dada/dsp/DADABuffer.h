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
#include "ipcio.h"

namespace dsp {

  //! Loads BitSeries data using DADA shared memory
  /*! This class pretends to be a file so that it can slip into the
    File::registry */
  class DADABuffer : public File
  {
    
  public:
    
    //! Constructor
    DADABuffer ();
    
    //! Construct given a shared memory I/O struct
    DADABuffer (const ipcio_t& _ipc);
    
    //! Destructor
    virtual ~DADABuffer () { }

    //! Returns true if filename appears to name a valid CPSR file
    bool is_valid (const char* filename,int NOT_USED=-1) const;

    //! Seek to the specified time sample
    virtual void seek (int64 offset, int whence = 0);

    //! Get the information about the data source
    virtual void set_info (const Observation& obs) { info = obs; }

    //! Reset DADAbuffer
    virtual void reset ();
 
  protected:

    //! Open the file
    virtual void open_file (const char* filename);

    //! Load bytes from shared memory
    virtual int64 load_bytes (unsigned char* buffer, uint64 bytes);
    
    //! Set the offset in shared memory
    virtual int64 seek_bytes (uint64 bytes);
   
    //! Shared memory interface
    ipcio_t ipc;
        
  };

}

#endif // !defined(__DADABuffer_h)
