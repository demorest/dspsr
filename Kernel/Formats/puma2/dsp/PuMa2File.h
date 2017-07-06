//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/puma2/dsp/PuMa2File.h


#ifndef __PuMa2File_h
#define __PuMa2File_h

#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from a PuMa2 data file
  class PuMa2File : public File 
  {

  public:
   
    //! Construct and open file
    PuMa2File (const char* filename=0);

    //! Returns true if filename appears to name a valid PuMa2 file
    bool is_valid (const char* filename) const;

    //! Set this to 'false' if you don't need to check bocf
    static bool want_to_check_bocf;

  protected:

    //! Open the file
    virtual void open_file (const char* filename);

    //! Read the PuMa2 ascii header from filename
    static std::string get_header (const char* filename);

  };

}

#endif // !defined(__PuMa2File_h)
  
