// CAPSRUnpackerSETUP.h


/*

 */

#ifndef __dsp_CASPSRUnpackerSetup_h
#define __dsp_CASPSRUnpackerSetup_h

#include<stdint.h>
#include "dsp/HistUnpacker.h"


namespace dsp {
  
  //

  class CASPSRUnpackerSetup : public HistUnpacker
  {
  public:

    //! Constructor
    CASPSRUnpackerSetup (const char* name = "CASPSRUpackerSetup");

    //! Destructor
    virtual ~CASPSRUnpackerSetup ();
    

  protected:
    
    void unpack();
    bool matches (const Observation*);
    BitSeries staging;

  };
}

#endif
