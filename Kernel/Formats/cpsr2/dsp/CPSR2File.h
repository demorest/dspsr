//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/cpsr2/dsp/CPSR2File.h

#ifndef __CPSR2File_h
#define __CPSR2File_h

#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from a CPSR2 data file
  class CPSR2File : public File 
  {
  public:
   
    //! Construct and open file
    CPSR2File (const char* filename=0);

    virtual ~CPSR2File();

    //! Returns true if filename appears to name a valid CPSR2 file
    bool is_valid (const char* filename) const;

    //! Set this to 'false' if you don't need to yamasaki verify
    static bool want_to_yamasaki_verify;

    //! return 'm' for cpsr1 and 'n' for cpsr2
    std::string get_prefix () const;

  protected:

    //! Pads gaps in data
    virtual int64_t pad_bytes(unsigned char* buffer, int64_t bytes);
      
    //! Open the file
    virtual void open_file (const char* filename);

    //! Read the CPSR2 ascii header from filename
    static int get_header (char* cpsr2_header, const char* filename);

    std::string prefix;
  };

}

#endif // !defined(__CPSR2File_h)
  
