//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/gmrt/dsp/GMRTBinaryFile.h


#ifndef __GMRTBinaryFile_h
#define __GMRTBinaryFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from a GMRTBinary data file
  class GMRTBinaryFile : public File 
  {

  public:
   
    //! Construct and open file
    GMRTBinaryFile (const char* filename=0);

    //! Returns true if filename appears to name a valid GMRTBinary file
    bool is_valid (const char* filename) const;

  protected:

    //! Open the file
    virtual void open_file (const char* filename);

    //! Read the ascii header from filename.hdr
    static std::string get_header (const char* filename);

  };

}

#endif // !defined(__GMRTBinaryFile_h)
  
