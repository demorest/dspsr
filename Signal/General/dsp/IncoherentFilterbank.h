//-*-C++-*-

#ifndef __IncoherentFilterbank_h
#define __IncoherentFilterbank_h

#include <memory>
#include <vector>

#include "genutil.h"
#include "Types.h"

#include "RealTimer.h"

#include "dsp/TimeSeries.h"
#include "dsp/TimeSeries.h"
#include "dsp/Transformation.h"

/*
NOTE: This transformation DESTROYS your input data

NOTE: According to WvS in his email of 14 January 2003 the FFT actually produces nchan+1 channels.  I have chosen to throw away the last (Nyquist) channel, to be consistent with dsp::Observation::get_base_frequency().  HSK

*/

namespace dsp{

  //  NOTE: This transformation DESTROYS your input data

  class IncoherentFilterbank : public Transformation<TimeSeries,TimeSeries>{

  public:

    //! Null constructor- operation is always out of place
    IncoherentFilterbank();

    //! Virtual Destructor
    virtual ~IncoherentFilterbank();
  
    //! Inquire transform size of current plan (zero=no plan)
    uint64 get_plansize(){ return wsave_size; }

    //! Free up the memory used by the current plan
    void free_plan(){ sink(wsave); wsave_size = 0; }
    
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

  protected:

    //! Perform the operation
    virtual void transformation ();

    //! Acquire the plan (wsave)
    void acquire_plan();

    //! Number of channels into which the input will be divided
    unsigned nchan;

    //! Memory used by MKL to store transform coefficients (ie the plan)
    auto_ptr<float> wsave; 
    
    //! The size of the memory allocated to wsave;
    uint64 wsave_size;

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

  };

}

#endif
