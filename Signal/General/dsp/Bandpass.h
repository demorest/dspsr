#ifndef __Bandpass_h
#define __Bandpass_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/Response.h"

namespace dsp {
  
  //! Produces the bandpass of an undetected timeseries.
  /*! This class produces a bandpass from an undetected timeseries
     it will eventually produce outputs equivalent in some respects to 
     those of the current response class. But no operation is performed
     upon this bandpass. If you want to perform operations in undetected
     frequency space use the Convolution class.


     the Bandpass::operate method simply produces the bandpass at a fixed
     time. So the output is a single timestep. You may think that this
    functionality could be incorporated into the filterbank class 
  */

  class Bandpass : public Transformation <TimeSeries,Response> {

  public:
    //! Null constructor
    Bandpass (const char* name = "Bandpass", Behaviour type = anyplace);
    //! Set the number of channels
    void set_nchan (unsigned _nchan) { nchan = _nchan; }

    //! Get the number of channels
    unsigned get_nchan () const { return nchan; }
    
    //! Verbosity flag
    static bool verbose;
    
    //! default destructor
    ~Bandpass ();

  protected:
    
    //! Perform the transformation on the input time series
    virtual void transformation ();

    //! Number of channels is passband
    unsigned nchan;

  };
}
#endif


