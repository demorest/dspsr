//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __WAPPFile_h
#define __WAPPFile_h

#include "dsp/BlockFile.h"

namespace dsp {

  //! Loads BitSeries data from a WAPP file
  class WAPPFile : public BlockFile
  {
  public:
	  
    //! Construct and open file	  
    WAPPFile (const char* filename=0);
	  
    //! Destructor
    ~WAPPFile ();
	  
    //! Returns true if filename is a valid WAPP file
    bool is_valid (const char* filename, int NOT_USED=-1) const;

  protected:

    //! Open the file
    void open_file (const char* filename);

    //! The header parser
    void* header;

  };

}

#endif // !defined(__WAPPFile_h)
