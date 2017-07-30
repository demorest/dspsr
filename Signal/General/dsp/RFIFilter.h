//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/RFIFilter.h

#ifndef __RFIFilter_h
#define __RFIFilter_h

#include "dsp/Response.h"
#include "Reference.h"
#include "MJD.h"

namespace dsp {

  class IOManager;
  class TimeSeries;
  class Bandpass;

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

    //! Set the buffer into which raw data will be read [optional]
    void set_buffer (TimeSeries* buffer);

    //! Set the buffer into which the spectra will be integrated [optional]
    void set_data (Response* data);

    //! Set the tool used to compute the spectra [optional]
    void set_bandpass (Bandpass* bandpass);

    void calculate (Response* bp);

  protected:

    //! The source of data
    Reference::To<IOManager> input;

    //! The buffer into which data will be read
    Reference::To<TimeSeries> buffer;

    //! The tool for computing the bandpass
    Reference::To<Bandpass> bandpass;

    //! The buffer into which the bandpass will be integrated
    Reference::To<Response> data;

    //! The maximum block size
    uint64_t maximum_block_size;

    //! The number of frequency channels used in calculating the bandpass
    unsigned nchan_bandpass;

    //! The interval over which the RFI mask will be calculated
    double interval;

    //! The fraction of the data used to calculate the RFI mask
    float duty_cycle;

    //! The size of the window used in median smoothing
    unsigned median_window;

  private:

    //! The start time of the epoch over which the RFI mask was calculated
    MJD end_time;

    //! The start time of the epoch over which the RFI mask was calculated
    MJD start_time;

    //! Set true when the response has been calculated over the interval
    bool calculated;

  };
  
}

#endif
