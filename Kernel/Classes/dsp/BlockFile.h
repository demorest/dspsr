//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/BlockFile.h


#ifndef __dsp_BlockFile_h
#define __dsp_BlockFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from a file organized in blocks
  class BlockFile : public File
  {
    
  public:
    
    //! Constructor
    BlockFile (const char* name);
    
    //! Virtual destructor
    virtual ~BlockFile ();

    //! Get the number of data bytes per block (frame)
    uint64_t get_block_data_bytes () const;

  protected:

    //! Total number of bytes in each block (header + data + tailer)
    uint64_t block_bytes;

    //! Number of bytes in header of each block
    uint64_t block_header_bytes;

    //! Number of bytes in tailer of each block
    uint64_t block_tailer_bytes;

    //! The current byte within a block
    uint64_t current_block_byte;

    //! Return ndat given the file and header sizes, nchan, npol, and ndim
    /*! Called by open_file for some file types, to determine that the
    header ndat matches the file size.  Requires 'info' parameters
    nchan, npol, and ndim as well as header_bytes to be correctly set */
    virtual int64_t fstat_file_ndat(uint64_t tailer_bytes=0);

    //! Load nbyte bytes of sampled data from the device into buffer
    /*! If the data stored on the device contains information other
      than the sampled data, this method should be overloaded and the
      additional information should be filtered out. */
    virtual int64_t load_bytes (unsigned char* buffer, uint64_t nbytes);

    //! Set the file pointer to the absolute number of sampled data bytes
    /*! If the header_bytes attribute is set, this number of bytes
      will be subtracted by File::seek_bytes before seeking.  If the
      data stored on the device after the header contains information
      other than the sampled data, this method should be overloaded
      and the additional information should be skipped. */
    virtual int64_t seek_bytes (uint64_t bytes);

    virtual void skip_extra ();
    
  private:

    //! Initialize variables to sensible null values
    void init();

  };

}

#endif // !defined(__File_h)
  

