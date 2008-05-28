//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __GMRTFile_h
#define __GMRTFile_h

#include "dsp/BlockFile.h"

namespace dsp {

  //! Loads BitSeries data from a MkV file
  class GMRTFile : public BlockFile
  {
  public:
	  
    //! Construct and open file	  
    GMRTFile (const char* filename=0, const char* headername=0);
	  
    //! Destructor
    ~GMRTFile ();
	  
    //! Returns true if filename is a valid Mk5 file
    bool is_valid (const char* filename) const;

  protected:

    //! Open the file
    void open_file (const char* filename);

    void skip_extra ();

  };

}

#endif // !defined(__GMRTFile_h)
