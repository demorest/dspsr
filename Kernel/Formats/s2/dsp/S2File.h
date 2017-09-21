//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/s2/dsp/S2File.h


#ifndef __S2File_h
#define __S2File_h

#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from a S2 data file.
  /*! The treatment of S2 data is specific to the use of S2-TCI at Swinburne */
  class S2File : public File 
  {
  public:
   
    //! Construct and open file
    S2File (const char* filename = 0);

    //! Returns true if filename appears to name a valid S2 file
    bool is_valid (const char* filename) const;

  protected:
    //! Open the file
    void open_file (const char* filename);
    
  private:
    
    //! Loads the extra S2 "filename.info" header file
    void load_S2info (const char* filename);

    //! Structure to be used with the load_S2info function
    typedef struct {
      std::string source;
      char telid;
      double freq;
      double calperiod;
      std::string tapeid;
    }S2_Extra_Hdr;

    S2_Extra_Hdr extra_hdr;
    
  };

}

#endif // !defined(__S2File_h)
  
