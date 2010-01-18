/*

 */

#ifndef __dsp_CASPSRUnpacker_h
#define __dsp_CASPSRUnpacker_h

namespace CUDA {
  
  //
  class CASPSRUnpackerGPU : public CASPSRUnpacker
  {
  public:

    //! Constructor
    CASPSRUnpackerGPU (const char* name = "CASPSRUpackerGPU");

    //! Destructor
    virtual ~CASPSRUnpacker ();
    

  protected:
    
    virtual void unpack ();


    //! staging buffer on the GPU for packed data
    float* stagingBufGPU;

    //! buffer for unpacked data on the GPU
    float* unpackBufGPU;

  };
}

#endif
