//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/Fold.h,v $
   $Revision: 1.16 $
   $Date: 2003/07/28 09:58:53 $
   $Author: wvanstra $ */


#ifndef __Fold_h
#define __Fold_h

#include <vector>

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/PhaseSeries.h"

class polyco;
class psrephem;

namespace dsp {

  class WeightedTimeSeries;

  //! Fold TimeSeries data into phase-averaged profile(s)
  /*! 
    This Operation does not modify the TimeSeries.  Rather, it accumulates
    the (folded) average pulse profile data within its data structures.
  */
  class Fold : public Transformation <const TimeSeries, PhaseSeries> {

  public:
    
    //! The maximum number of phase bins returned by Fold::choose_nbin
    static unsigned maximum_nbin;

    //! The minimum width of each pulse phase bin; used by Fold::choose_nbin
    static double dsp::Fold::minimum_bin_width;

    //! Controls the number of phase bins returned by Fold::choose_nbin
    static bool power_of_two;

    //! Constructor
    Fold ();
    
    //! Destructor
    virtual ~Fold ();

    //! Prepare to fold the input TimeSeries
    void prepare ();

    //! Prepare to fold the given Observation
    void prepare (const Observation* observation);

    //! Set the number of phase bins into which data will be folded
    void set_nbin (unsigned _nbin) { requested_nbin = _nbin; }
    //! Get the number of phase bins into which data will be folded
    unsigned get_nbin () const { return requested_nbin; }

    //! Set the number of polynomial coefficients in model
    void set_ncoef (unsigned ncoef);
    //! Get the number of polynomial coefficients in model
    unsigned get_ncoef () const { return ncoef; }

    //! Set the number of minutes over which polynomial coefficients are valid
    void set_nspan (unsigned nspan);
    //! Get the number of minutes over which polynomial coefficients are valid
    unsigned get_nspan () const { return nspan; }

    //! Set the period at which to fold data (in seconds)
    void set_folding_period (double folding_period);
    //! Get the period at which to fold data (in seconds)
    double get_folding_period () const;

    //! Set the phase model with which to fold data
    void set_folding_polyco (const polyco* folding_polyco);
    //! Get the phase model with which to fold data
    const polyco* get_folding_polyco () const;

    //! Set the ephemeris with which to create the phase model
    void set_pulsar_ephemeris (const psrephem* pulsar_ephemeris);
    //! Get the ephemeris with which to create the phase model
    const psrephem* get_pulsar_ephemeris () const;

    //! Overload Transformation::set_input to set weighted_input
    void set_input (TimeSeries* input);

    //! Add a phase model with which to choose to fold the data
    void add_folding_polyco (polyco* folding_polyco);

    //! Add an ephemeris with which to choose to create the phase model
    void add_pulsar_ephemeris (psrephem* pulsar_ephemeris);

    //! Choose an appropriate ephemeris from those added
    psrephem* choose_ephemeris (const string& pulsar);

    //! Choose an appropriate polyco from those added
    polyco* choose_polyco (const MJD& time, const string& pulsar);

    //! Choose an appropriate number of pulse phase bins
    unsigned choose_nbin ();

    //! Fold nblock blocks of data
    void fold (double& integration_length, float* phase, unsigned* hits,
	       const Observation* info, unsigned nblock,
	       const float* time, uint64 ndat, unsigned ndim,
	       const unsigned* weights=0, unsigned ndatperweight=0,
	       uint64 idat_start=0, uint64 ndat_fold=0);


  protected:

    //! The transformation folds the data into the profile
    virtual void transformation ();

    //! Period at which to fold data (CAL)
    double folding_period;

    //! Phase model with which to fold data (PSR)
    Reference::To<const polyco> folding_polyco;

    //! Ephemeris with which to create the phase model
    Reference::To<const psrephem> pulsar_ephemeris;

    //! Set when Tranformation::input is a Weighted TimeSeries
    Reference::To<const WeightedTimeSeries> weighted_input;

    //! Number of phase bins into which the data will be integrated
    unsigned folding_nbin;

    //! Number of phase bins set using set_nbin
    unsigned requested_nbin;

    //! Number of polynomial coefficients in model
    unsigned ncoef;

    //! Number of minutes over which polynomial coefficients are valid
    unsigned nspan;

    //! Flag that the polyco is built for the given ephemeris and input
    bool built;

    //! Polycos from which to choose
    vector< Reference::To<polyco> > polycos;

    //! Ephemerides from which to choose
    vector< Reference::To<psrephem> > ephemerides;

    //! INTERNAL: the time sample at which to start folding
    uint64 idat_start;

    //! INTERNAL: the number of time samples to fold
    uint64 ndat_fold;

  };
}

#endif // !defined(__Fold_h)


