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

    //! staging buffer on the GPU for packed data
    unsigned char* stagingBufGPU;

    //! buffer for unpacked data on the GPU
    float* unpackBufGPU;

    unsigned char* host_mem;

  };
}

#endif
