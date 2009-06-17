//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2003 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __IncoherentFilterbank_h
#define __IncoherentFilterbank_h

#include "dsp/TimeSeries.h"
#include "dsp/Transformation.h"

#include "RealTimer.h"

/*

NOTE: According to WvS in his email of 14 January 2003 the FFT
actually produces nchan+1 channels.  I have chosen to throw away the
last (Nyquist) channel, to be consistent with
dsp::Observation::get_base_frequency().  HSK

*/

namespace dsp{

  class IncoherentFilterbank : public Transformation<TimeSeries,TimeSeries>{

  public:

    //! Null constructor- operation is always out of place
    IncoherentFilterbank();

    //! Virtual Destructor
    virtual ~IncoherentFilterbank();
  
    //! Inquire transform size of current plan (zero=no plan)
    uint64_t get_plansize(){ return wsave_size; }

    //! Free up the memory used by the current plan
    void free_plan(){ if(wsave) delete [] wsave; wsave = 0; wsave_size = 0; }
    
    //! Set the number of channels
    void set_nchan(unsigned _nchan){ nchan = _nchan; }

    //! Inquire the number of channels in the filterbank
    unsigned get_nchan(){ return nchan; }

    //! Set the output state- one of Intensity, PPQQ, Analytic
    void set_output_state(Signal::State _state){ state = _state; }

    //! Inquire the output state- default is Intensity
    Signal::State get_output_state(){ return state; }

    //! Inquire timings
    double get_fft_time(){ return fft_timer.get_total(); }
    double get_fft_loop_time(){ return fft_loop_timer.get_total(); }
    double get_conversion_time(){ return conversion_timer.get_total(); }

    //! Inquire loop unroll level for forming incoherent filterbank
    unsigned get_unroll_level(){ return unroll_level; }
    
    //! Set loop unroll level for forming incoherent filterbank
    void set_unroll_level(unsigned _unroll_level){ unroll_level = _unroll_level; }

    //! Inquire whether input can be destroyed upon call to operate()
    bool get_destroy_input(){ return destroy_input; }
    
    //! Set whether input can be destroyed upon call to operate()
    void set_destroy_input(bool _destroy_input){ destroy_input = _destroy_input; }

  protected:

    //! Perform the operation
    virtual void transformation ();

    //! Number of channels into which the input will be divided
    unsigned nchan;

    //! Memory used by MKL to store transform coefficients (ie the plan)
    float* wsave; 
    
    //! The size of the memory allocated to wsave;
    uint64_t wsave_size;

    //! The output's state (ie the number of polarisations)
    Signal::State state;

    //! Worker function for state=Signal::Intensity
    virtual void form_stokesI();

    //! Worker function for state=Signal::PPQQ
    virtual void form_PPQQ();

    //! Worker function for state=Signal::Analytic
    virtual void form_undetected();

    //! Timer for FFT'ing
    RealTimer fft_timer;

    //! Timer for FFT loop
    RealTimer fft_loop_timer;

    //! Timer for TimeSeries conversion
    RealTimer conversion_timer;

    //! Loop unroll level for forming undetected filterbank
    unsigned unroll_level;

    //! If set to true, the input data array may be destroyed on the call to operation()
    bool destroy_input;

  };

}

#endif


