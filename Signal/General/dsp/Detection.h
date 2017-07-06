//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/Detection.h


#ifndef __Detection_h
#define __Detection_h

class Detection;

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  //! Detects phase-coherent TimeSeries data
  /*!  

  The Detection class may be used to perform simple square law
  detection or calculation of the Stokes parameters or the coherency
  matrix.  In the case of Stokes/Coherency formation, the components
  may be stored in polarization-major order or time-major order, or a
  mixture of the two by calling set_output_ndim() with an argument of
  4, 1, or 2, respectively.  The three methods require different
  amounts of RAM and therefore result in performance benefits that are
  largely cache-dependent.

  It is recommended that both set_output_ndim and set_output_state 
  are called.

  */
  class Detection : public Transformation <TimeSeries, TimeSeries> {

  public:
    
    //! Constructor
    Detection ();
    
    //! Prepare the output TimeSeries attributes
    void prepare ();

    //! Set the state of the output data
    void set_output_state (Signal::State _state);
    //! Get the state of the output data
    Signal::State get_output_state () const { return state; }

    //! Set the dimension of the output data
    void set_output_ndim (int _ndim) { ndim = _ndim; }
    //! Get the dimension of the output data
    bool get_output_ndim () const { return ndim; }

    //! Return true if the specified input data order can be supported
    bool get_order_supported (TimeSeries::Order) const;

    //! Engine used to perform discrete convolution step
    class Engine;
    void set_engine (Engine*);
    
  protected:

    //! Detect the input data
    virtual void transformation ();

    //! Signal::State of the output data
    Signal::State state;

    //! Dimension of the output data
    int ndim;

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

    //! Called by polarimetry to return pointers to the result channels
    void get_result_pointers (unsigned ichan, bool inplace, float* r[4]);

    //! Perform simple square-law detection
    void square_law ();

    //! Polarization detection (Stokes parameters or Coherency products)
    void polarimetry ();

    //! Set the state of the output TimeSeries
    void resize_output ();

    //! Throws an Error if something is wrong
    void checks();

    //! Quick and dirty method for detecting to PP or QQ
    void onepol_detect();
  };

  class Detection::Engine : public Reference::Able
  {
  public:
    virtual void polarimetry (unsigned ndim,
			      const TimeSeries* in, TimeSeries* out) = 0;

    virtual void square_law (const dsp::TimeSeries* input,
            dsp::TimeSeries* output) = 0;
  }; 
}

#endif // !defined(__Detection_h)
