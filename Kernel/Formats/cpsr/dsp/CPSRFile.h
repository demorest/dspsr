//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/dsp/CPSRFile.h,v $
   $Revision: 1.9 $
   $Date: 2003/02/13 01:35:08 $
   $Author: pulsar $ */


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

    //! Returns true if filename appears to name a valid CPSR file
    bool is_valid (const char* filename) const;

    //! The tape number
    int tapenum;

    //! The file number on tape
    int filenum;
    
    //! Return a pointer to an possibly identical instance of a CPSRFile
    virtual CPSRFile* clone(bool identical=true);

  protected:
    //! Open the file
    virtual void open_file (const char* filename,PseudoFile* _info);

  };

}

#endif // !defined(__CPSRFile_h)
  
