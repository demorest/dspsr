//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Aidan Hotan
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/maxim/dsp/MaximFile.h,v $
   $Revision: 1.2 $
   $Date: 2006/07/09 13:27:08 $
   $Author: wvanstra $ */

#ifndef __MaximFile_h
#define __MaximFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads TimeSeries data from a Maxim data file.
  class MaximFile : public File 
  {
  public:
    
    //! Construct and open file
    MaximFile (const char* filename = 0);
    
    //! Destructor
    ~MaximFile ();

    //! Return true if filename contains data in the recognized format.
    bool is_valid (const char *filename,int NOT_USED=-1) const;
    
    
  protected:

    //! Open the file
    void open_file (const char* filename);
    
    
  private:
    
    //! Holds the start time for the data file.
    char timestamp[17]; 
    
  };
}


#endif // __MaximFile_h
