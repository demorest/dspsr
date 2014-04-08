//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011, 2013 by James M Anderson  (MPIfR)
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/



#ifndef __LuMPFile_h
#define __LuMPFile_h

#include "dsp/File.h"
#include "dsp/LuMPObservation.h"

namespace dsp {

  //! Loads BitSeries data from a LuMP data file
  class LuMPFile : public File 
  {

  public:

    //! Construct and open file
    LuMPFile (const char* filename=0);

    //! Returns true if filename appears to name a valid LuMP file
    bool is_valid (const char* filename) const;

    //! Get the information about the data source
    virtual LuMPObservation* get_lump_info () { return dynamic_cast<LuMPObservation*>(get_info()); }

    //! Get the information about the data source
    virtual const LuMPObservation* get_lump_info () const { return dynamic_cast<const LuMPObservation*>(get_info()); }

  protected:

    friend class LuMPUnpacker;

    //! Open the file
    virtual void open_file (const char* filename);

    //! Load bytes from file
    virtual int64_t load_bytes (unsigned char* buffer, uint64_t bytes);
    
    //! Adjust the file pointer
    virtual int64_t seek_bytes (uint64_t bytes);
      
    //! Return ndat given the file and header sizes, nchan, npol, and ndim
    /*! Called by open_file for some file types, to determine that the
    header ndat matches the file size.  Requires 'info' parameters
    nchan, npol, and ndim as well as header_bytes to be correctly set */
    virtual int64_t fstat_file_ndat(uint64_t tailer_bytes=0);
    
    //! Read the LuMP ascii header from filename
    static std::string get_header (const char* filename);

    //! Current data location in file, in units of bytes
    /*! Note that for a regular file, this is an offset from header_bytes
    bytes into the file */
    int64_t file_data_position;
  };

}

#endif // !defined(__LuMPFile_h)
  
