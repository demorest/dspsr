//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/cpsr/dsp/CPSRFile.h


#ifndef __CPSRFile_h
#define __CPSRFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from a CPSR data file
  class CPSRFile : public File 
  {
  public:

    //! Construct and open file
    CPSRFile (const char* filename = 0);

    //! Virtual destructor
    virtual ~CPSRFile();

    //! Returns true if filename appears to name a valid CPSR file
    bool is_valid (const char* filename) const;

    //! The tape number
    int tapenum;

    //! The file number on tape
    int filenum;
    
  protected:
    //! Open the file
    virtual void open_file (const char* filename);

  };

}

#endif // !defined(__CPSRFile_h)
  
