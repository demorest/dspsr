//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/spigot/dsp/SpigotFile.h,v $
   $Revision: 1.2 $
   $Date: 2004/01/23 18:25:39 $
   $Author: wvanstra $ */


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
    bool is_valid (const char* filename,int NOT_USED=-1) const;

  protected:

    //! Open the file
    void open_file (const char* filename);

    //! Set the Input::info attribute, given a loaded Header_type object
    void parse (void* hdr);

  };

}

#endif // !defined(__SpigotFile_h)
  
