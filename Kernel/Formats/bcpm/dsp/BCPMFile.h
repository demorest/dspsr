//-*-C++-*-

#ifndef __dsp_BCPMFile_h
#define __dsp_BCPMFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from a BCPM data file.
  class BCPMFile : public File 
  {

  public:
   
    //! Construct and open file
    BCPMFile (const char* filename = 0);

    //! Destructor
    ~BCPMFile ();

    //! Returns true if filename appears to name a valid BCPM file
    bool is_valid (const char* filename,int NOT_USED=-1) const;

  protected:

    //! Open the file
    void open_file (const char* filename);

  };

}

#endif // !defined(__BCPMFile_h)
  
