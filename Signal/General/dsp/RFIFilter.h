//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/RFIFilter.h,v $
   $Revision: 1.1 $
   $Date: 2004/10/18 14:13:31 $
   $Author: wvanstra $ */

#ifndef __RFIFilter_h
#define __RFIFilter_h

#include "dsp/Response.h"
#include "Reference.h"
#include "MJD.h"

namespace dsp {

  class IOManager;
  class TimeSeries;

  //! Real-time RFI mitigation using a frequency response function
  /* This class may become a base class for all RFI mitigaton
     strategies that can be applied in the frequency domain.  For now,
     it produces a narrow-band birdie zapping mask. */
  class RFIFilter: public Response {

  public:

    //! Default constructor
    RFIFilter ();

    //! Destructor
    ~RFIFilter ();

    //! Create an RFI filter for the specified observation
    void match (const Observation* input, unsigned nchan);

    //! Create an RFI filter with the same number of channels as Response
    void match (const Response* response);

    //! Set the number of channels into which the band will be divided
    void set_nchan (unsigned nchan);

    //! Set the interval over which the RFI mask will be calculated
    void set_update_interval (double seconds);

    //! Set the fraction of the data used to calculate the RFI mask
    void set_duty_cycle (float cycle);

    //! Set the source of the data
    void set_input (IOManager* input);

    //! Set the buffer into which data will be read
    void set_buffer (TimeSeries* buffer);

  protected:

    //! The source of data
    Reference::To<IOManager> input;

    //! The buffer into which data will be read
    Reference::To<TimeSeries> buffer;

    //! The maximum block size
    uint64 maximum_block_size;

    //! The number of frequency channels chosen by user
    unsigned nchan_requested;

    //! The interval over which the RFI mask will be calculated
    double interval;

    //! The fraction of the data used to calculate the RFI mask
    float duty_cycle;

  private:

    //! The start time of the epoch over which the RFI mask was calculated
    MJD end_time;

    //! The start time of the epoch over which the RFI mask was calculated
    MJD start_time;

  };
  
}

#endif
