//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/dsp/CPSRFile.h,v $
   $Revision: 1.12 $
   $Date: 2004/12/12 05:50:17 $
   $Author: hknight $ */


#ifndef __CPSRFile_h
#define __CPSRFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads TimeSeries data from a CPSR data file
  class CPSRFile : public File 
  {
  public:

    //! Construct and open file
    CPSRFile (const char* filename = 0);

    //! Virtual destructor
    virtual ~CPSRFile();

    //! Returns true if filename appears to name a valid CPSR file
    bool is_valid (const char* filename,int NOT_USED=-1) const;

    //! The tape number
    int tapenum;

    //! The file number on tape
    int filenum;
    
  protected:
    //! Open the file
    virtual void open_file (const char* filename);

  };

}

#endif // !defined(__CPSRFile_h)
  
