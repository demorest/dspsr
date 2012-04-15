//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __LOFAR_DALFile_h
#define __LOFAR_DALFile_h

#include "dsp/File.h"

namespace dsp {

  //! Wrapper around the LOFAR DAL classes used to load beam-formed data
  class LOFAR_DALFile : public File
  {
  public:
	  
    //! Construct and open file	  
    LOFAR_DALFile (const char* project_id=0);
	  
    //! Destructor
    ~LOFAR_DALFile ();
	  
    //! Returns true if filename is a valid LOFAR beam-formed file
    bool is_valid (const char* filename) const;

  protected:

    void open_file (const char* filename);  

    void close ();
    void rewind ();
    int64_t load_bytes (unsigned char* buffer, uint64_t bytes);
    int64_t seek_bytes (uint64_t bytes);

    class Handle;
    Handle* handle;

  };

}

#endif // !defined(__LOFAR_DALFile_h)
