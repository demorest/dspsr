//-*-C++-*-

#ifndef __IncoherentFilterbank_h
#define __IncoherentFilterbank_h

#include <memory>
#include <vector>

#include "genutil.h"

#include "dsp/TimeSeries.h"
#include "dsp/TimeSeries.h"
#include "dsp/Transformation.h"

/* The IncoherentFilterbank is designed for searching and thus square law detects + pscrunches the data.  If you want to form Stokes parameters then it is suggested that you work out how you do it and then implement it.

NOTE: This transformation DESTROYS your input data

NOTE: According to WvS in his email of 14 January 2003 the FFT actually produces nchan+1 channels.  I have chosen to throw away the last (Nyquist) channel, to be consistent with dsp::Observation::get_base_frequency().  I don't actually understand it myself.  HSK 16/1/03

*/

namespace dsp{

  class IncoherentFilterbank : public Transformation<TimeSeries,TimeSeries>{

  public:

    //! Null constructor- operation is always out of place
    IncoherentFilterbank();

    //! Virtual Destructor
    virtual ~IncoherentFilterbank();
  
    //! Inquire transform size of current plan (zero=no plan)
    unsigned get_plansize(){ if(!wsave.get()) return 0; return (wsave->size()-4)/2; }

    //! Free up the memory used by the current plan
    void free_plan(){ sink(wsave); }
    
    //! Set the number of channels
    void set_nchan(unsigned _nchan){ nchan = _nchan; }

    //! Inquire the number of channels in the filterbank
    unsigned get_nchan(){ return nchan; }

  protected:

    //! Perform the operation
    virtual void transformation ();

    //! Acquire the plan (wsave)
    void acquire_plan();

    //! Number of channels into which the input will be divided
    unsigned nchan;

    //! Memory used by MKL to store transform coefficients (ie the plan)
    auto_ptr<vector<float> > wsave; 

  };

}

#endif
