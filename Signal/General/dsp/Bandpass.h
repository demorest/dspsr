/***************************************************************************
 *
 *   Copyright (C) 2002 by Stephen Ord
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#ifndef __dsp_Bandpass_h
#define __dsp_Bandpass_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/Response.h"

namespace dsp {
  
  class Apodization;

  //! Produces the bandpass of an undetected timeseries.
  class Bandpass : public Transformation<TimeSeries, Response> {

  public:

    //! Default constructor
    Bandpass ();

    //! Destructor
    ~Bandpass ();

    //! Set the number of frequency channels in the bandpass
    void set_nchan (unsigned nchan) { resolution = nchan; }
    //! Get the number of frequency channels in the bandpass
    unsigned get_nchan () const { return resolution; }
    
    //! Set the state of the output
    void set_state (Signal::State state) { output_state = state; }
    //! Get the state of the output
    Signal::State get_state () const { return output_state; }

    //! Set the frequency response function
    virtual void set_response (Response* response);

    //! Return true if the response attribute has been set
    bool has_response () const;

    //! Return a pointer to the frequency response function
    virtual const Response* get_response() const;

    //! Set the apodization function
    virtual void set_apodization (Apodization* function);

    //! Get the integration length (in seconds)
    double get_integration_length() { return integration_length; }

    //! Set the integration length and bandpass to zero
    void reset_output();

  protected:
    
    //! Perform the transformation on the input time series
    void transformation ();
    void detected_input ();

    //! Number of channels in bandpass
    unsigned resolution;

    //! Integration length in seconds
    double integration_length;

    //! Output state
    Signal::State output_state;

    //! Apodization function (time domain window)
    Reference::To<Apodization> apodization;

    //! Frequency response (fractional delay and fringe, for example)
    Reference::To<Response> response;
  };

}

#endif


