//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/spigot/dsp/SpigotFile.h,v $
   $Revision: 1.1 $
   $Date: 2004/01/23 16:12:17 $
   $Author: wvanstra $ */


#ifndef __dsp_PuMaFile_h
#define __dsp_PuMaFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads TimeSeries data from a PuMa data file.
  /*! This class is heavily copied from PuMaData by Russell Edwards. */
  class PuMaFile : public File 
  {
  public:
   
    //! Construct and open file
    PuMaFile (const char* filename = 0);

    //! Destructor
    ~PuMaFile ();

    //! Returns true if filename appears to name a valid PuMa file
    bool is_valid (const char* filename,int NOT_USED=-1) const;

  protected:

    //! Open the file
    void open_file (const char* filename);

    //! Set the Input::info attribute, given a loaded Header_type object
    void parse (const void* hdr);

  private:

    //! The PuMa header
    void* header;

  };

}

#endif // !defined(__PuMaFile_h)
  
