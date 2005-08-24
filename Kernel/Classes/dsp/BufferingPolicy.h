//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/BufferingPolicy.h,v $
   $Revision: 1.3 $
   $Date: 2005/08/24 05:54:21 $
   $Author: hknight $ */

#ifndef __baseband_dsp_BufferingPolicy_h
#define __baseband_dsp_BufferingPolicy_h

#include <string>

#include "psr_cpp.h"
#include "ReferenceAble.h"

namespace dsp {

  //! Defines the interface by which Transformation data are buffered
  /*! This pure virtual base class defines the interface by which
    Transformation input and/or output may be buffered. */
  class BufferingPolicy : public Reference::Able {
    
  public:
    
    //! Perform all buffering tasks required before transformation
    virtual void pre_transformation () = 0;
    
    //! Perform all buffering tasks required after transformation
    virtual void post_transformation () = 0;

    //! Set the first sample to be used from the input next time
    virtual void set_next_start (uint64 next_start_sample) = 0;

    //! Set the minimum number of samples that can be processed
    virtual void set_minimum_samples (uint64 minimum_samples) = 0;

    //! Returns the name
    string get_name(){ return name; }

  protected:

    //! Descriptive name
    string name;

  };

}

#endif
