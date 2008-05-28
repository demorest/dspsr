//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __WAPPFile_h
#define __WAPPFile_h

#include "dsp/File.h"


namespace dsp {

  class WAPPUnpacker;

  //! Loads BitSeries data from a WAPP file
  class WAPPFile : public File
  {
  public:
	  
    //! Construct and open file	  
    WAPPFile (const char* filename=0);
	  
    //! Destructor
    ~WAPPFile ();
	  
    //! Returns true if filename is a valid WAPP file
    bool is_valid (const char* filename) const;

  protected:

    friend class WAPPUnpacker;

    //! Open the file
    void open_file (const char* filename);

    //! The header parser
    void* header;

  };

}

#endif // !defined(__WAPPFile_h)
