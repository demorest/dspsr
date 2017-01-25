//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/spigot/dsp/SpigotFile.h


#ifndef __dsp_SpigotFile_h
#define __dsp_SpigotFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from a Spigot data file.
  class SpigotFile : public File 
  {

  public:
   
    //! Construct and open file
    SpigotFile (const char* filename = 0);

    //! Destructor
    ~SpigotFile ();

    //! Returns true if filename appears to name a valid Spigot file
    bool is_valid (const char* filename) const;

  protected:

    //! Open the file
    void open_file (const char* filename);

    //! Set the Input::info attribute, given a loaded Header_type object
    void parse (void* hdr);

  };

}

#endif // !defined(__SpigotFile_h)
  
