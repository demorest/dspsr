//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/lbadr/dsp/SMROFile.h,v $
   $Revision: 1.1 $
   $Date: 2004/06/09 02:02:41 $
   $Author: cwest $ */

#ifndef __SMROFile_h
#define __SMROFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads TimeSeries data from a SMRO data file.
  class SMROFile : public File 
  {
  public:
    
    //! Construct and open file
    SMROFile (const char* filename = 0);
    
    //! Destructor
    ~SMROFile ();

    //! Return true if filename contains data in the recognized format.
    bool is_valid (const char *filename,int NOT_USED=-1) const;
    
    
  protected:

    //! Open the file
    void open_file (const char* filename);
    
    
  private:
    
    //! Holds the start time for the data file.
    //! Note: This is TEMPORARY and will contact a full header in the future.
    char timestamp[17]; 
    
  };
}


#endif // __SMROFile_h
