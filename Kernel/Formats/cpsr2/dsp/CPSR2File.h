//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr2/dsp/CPSR2File.h,v $
   $Revision: 1.10 $
   $Date: 2003/03/14 00:59:43 $
   $Author: pulsar $ */


#ifndef __CPSR2File_h
#define __CPSR2File_h

#include "dsp/File.h"

namespace dsp {

  //! Loads TimeSeries data from a CPSR2 data file
  class CPSR2File : public File 
  {
  public:
   
    //! Construct and open file
    CPSR2File (const char* filename=0);

    //! Returns true if filename appears to name a valid CPSR2 file
    bool is_valid (const char* filename) const;

    //! Return a pointer to an possibly identical instance of a CPSR2File
    virtual CPSR2File* clone(bool identical=true);

    //! Set this to 'false' if you don't need to yamasaki verify
    static bool want_to_yamasaki_verify;
    
  protected:

    //! Open the file
    virtual void open_file (const char* filename,PseudoFile* _info);

    //! Read the CPSR2 ascii header from filename
    static int get_header (char* cpsr2_header, const char* filename);
  };

}

#endif // !defined(__CPSR2File_h)
  
