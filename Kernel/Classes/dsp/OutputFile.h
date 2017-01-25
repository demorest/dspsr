//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/OutputFile.h

#ifndef __OutputFile_h
#define __OutputFile_h

#include "Error.h"
#include "dsp/HasInput.h"
#include "dsp/Operation.h"
#include "dsp/BitSeries.h"

namespace dsp {

  //! Pure virtual base class of all objects that can unload BitSeries data
  /*! 
    This class defines the common interface as well as some basic
    functionality relating to sources of BitSeries data.
  */

  class OutputFile : public Operation, public HasInput<BitSeries>
  {
  public:
    
    //! Constructor
    OutputFile (const char* operation_name);
    
    //! Destructor
    virtual ~OutputFile ();

  protected:

    friend class OutputFileShare;

    //! Unload data into the BitSeries specified with set_output
    virtual void operation ();

    //! The file descriptor
    int fd;
    
    //! The size of the header in bytes
    int header_bytes;

    //! The name of the output file
    std::string output_filename;

    //! The pattern used to create an output filename
    std::string datestr_pattern;

    //! Open the file specified by filename for writing
    virtual void open_file (const char* filename);  
    //! Convenience wrapper
    void open_file (const std::string& name) { open_file (name.c_str()); }

    //! Write the file header to the open file
    virtual void write_header () = 0;

    //! Get the extension to be added to the end of new filenames
    virtual std::string get_extension () const = 0;

    //! Load nbyte bytes of sampled data from the device into buffer
    virtual int64_t unload_bytes (const void* buffer, uint64_t nbytes);
  };

}

#endif // !defined(__OutputFile_h)
