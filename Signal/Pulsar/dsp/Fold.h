//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/Fold.h,v $
   $Revision: 1.4 $
   $Date: 2002/10/08 17:07:06 $
   $Author: wvanstra $ */


#ifndef __Fold_h
#define __Fold_h

#include <vector>

#include "Operation.h"
#include "environ.h"
#include "MJD.h"

class polyco;
class polynomial;

namespace dsp {

  class PhaseSeries;

  //! Fold Timeseries data into phase-averaged profile(s)
  /*! 
    This Operation does not modify the Timeseries.  Rather, it accumulates
    the (folded) average pulse profile data within its data structures.
  */
  class Fold : public Operation {

  public:
    
    //! Constructor
    Fold ();
    
    //! Destructor
    ~Fold ();

    //! Set the number of phase bins into which data will be folded
    void set_nbin (unsigned _nbin) { nbin = _nbin; }
    //! Set the number of phase bins into which data will be folded
    unsigned get_nbin () const { return nbin; }

    //! Set the period at which to fold data (in seconds)
    void set_folding_period (double folding_period);
    //! Get the period at which to fold data (in seconds)
    double get_folding_period () const;

    //! Set the phase polynomial(s) with which to fold data
    void set_folding_polyco (polyco* folding_polyco);
    //! Get the phase polynomial(s) with which to fold data
    polyco* get_folding_polyco () const;

    //! Time of first sample to be folded
    void set_start_time (MJD _start_time)
    { start_time = _start_time; }

    //! Sampling interval of data to be folded
    void set_sampling_interval (double _sampling_interval)
    { sampling_interval = _sampling_interval; }

    //! Fold nblock blocks of data
    void fold (unsigned nblock, int64 ndat, unsigned ndim,
	       const float* time, float* phase, unsigned* hits,
	       int64 ndat_fold=0);

  protected:

    //! The operation folds the data into the profile
    virtual void operation ();

    //! Period at which to fold data (CAL)
    double folding_period;

    //! Phase polynomial(s) with which to fold data (PSR)
    Reference::To<polyco> folding_polyco;

    //! Number of phase bins into which the data will be integrated
    unsigned nbin;

    //! Time of first sample to be folded
    MJD start_time;

    //! Sampling interval of data to be folded
    double sampling_interval;

  };
}

#endif // !defined(__Fold_h)
