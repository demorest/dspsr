//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/BufferingPolicy.h

#ifndef __baseband_dsp_BufferingPolicy_h
#define __baseband_dsp_BufferingPolicy_h

#include "OwnStream.h"
#include "environ.h"

#include <string>

namespace dsp {

  //! Defines the interface by which Transformation data are buffered
  /*! This pure virtual base class defines the interface by which
    Transformation input and/or output may be buffered. */
  class BufferingPolicy : public OwnStream {
    
  public:
    
    //! Perform all buffering tasks required before transformation
    virtual void pre_transformation () = 0;
    
    //! Perform all buffering tasks required after transformation
    virtual void post_transformation () = 0;

    //! Set the first sample to be used from the input next time
    virtual void set_next_start (uint64_t next_start_sample) = 0;

    //! Set the minimum number of samples that can be processed
    virtual void set_minimum_samples (uint64_t minimum_samples) = 0;

    //! Returns the name
    std::string get_name() { return name; }

  protected:

    //! Descriptive name
    std::string name;

  };

}

#endif
