//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/Fold.h,v $
   $Revision: 1.60 $
   $Date: 2010/04/23 01:49:22 $
   $Author: straten $ */

#ifndef __baseband_dsp_Fold_h
#define __baseband_dsp_Fold_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/PhaseSeries.h"

namespace Pulsar {
  class Predictor;
  class Parameters;
}

namespace dsp {

  class WeightedTimeSeries;
  class ObservationChange;

  //! Fold TimeSeries data into phase-averaged profile(s)
  /*! 
    This Operation does not modify the TimeSeries.  Rather, it accumulates
    the (folded) average pulse profile data within its data structures.
  */
  class Fold : public Transformation <TimeSeries, PhaseSeries> {

    friend class MultiFold;
    friend class rawprofile;

  public:
    
    //! The maximum number of phase bins returned by Fold::choose_nbin
    static unsigned maximum_nbin;

    //! The minimum width of each pulse phase bin; used by Fold::choose_nbin
    static double minimum_bin_width;

    //! Controls the number of phase bins returned by Fold::choose_nbin
    static bool power_of_two;

    //! Constructor
    Fold ();
    
    //! Destructor
    ~Fold ();

    //! Create a clonse
    Fold* clone () const;

    //! Prepare to fold the input TimeSeries
    void prepare ();

    //! Prepare to fold the given Observation
    void prepare (const Observation* observation);

    //! If Operation is a Fold, integrate its PhaseSeries
    void combine (const Operation*);

    //! Reset the PhaseSeries
    void reset ();

    //! Perform any final operations
    virtual void finish ();

    //! Set the number of phase bins into which data will be folded
    void set_nbin (unsigned _nbin) { requested_nbin = _nbin; }
    //! Get the number of phase bins into which data will be folded
    unsigned get_nbin () const { return requested_nbin; }

    void set_force_sensible_nbin (bool _force_sensible) { force_sensible_nbin = _force_sensible; }
    bool get_force_sensible_nbin () { return force_sensible_nbin; }

    //! Set the number of polynomial coefficients in model
    void set_ncoef (unsigned ncoef);
    //! Get the number of polynomial coefficients in model
    unsigned get_ncoef () const { return ncoef; }

    //! Set the number of minutes over which polynomial coefficients are valid
    void set_nspan (unsigned nspan);
    //! Get the number of minutes over which polynomial coefficients are valid
    unsigned get_nspan () const { return nspan; }

    //! Set the attributes to be changed in the output PhaseSeries
    void set_change (const ObservationChange*);

    //! Set the period at which to fold data for all sources 
    /*! \param folding_period in seconds, negative to disable */
    void set_folding_period (double folding_period);

    //! Set the period at which to fold data for the specified source
    void set_folding_period (double folding_period, std::string source);

    //! Get the period at which data is being folded (in seconds)
    double get_folding_period () const;

    //! Returns true if data will be folded at a constant period
    bool has_folding_period() const { return (folding_period>0); }

    //! Get the phase model which is currently being used to fold data
    const Pulsar::Predictor* get_folding_predictor () const;

    //! Returns true if data will be folded using Pulsar::Predictor
    bool has_folding_predictor () const;

    //! Get the ephemeris used to create the phase model
    const Pulsar::Parameters* get_pulsar_ephemeris () const;

    //! Set the phase model with which to fold data
    void set_folding_predictor (const Pulsar::Predictor*);

    //! use to see whether pulsar eph is stored
    bool has_pulsar_ephemeris() const { return pulsar_ephemeris.ptr(); }

    //! Set the ephemeris with which to create a new phase model
    void set_pulsar_ephemeris (const Pulsar::Parameters*);

    //! Set the reference phase (phase of bin zero)
    void set_reference_phase (double phase);
    //! Get the reference phase (phase of bin zero)
    double get_reference_phase () const { return reference_phase; }

    //! Overload Transformation::set_input to set weighted_input
    void set_input (const TimeSeries* input);

    //! Choose an appropriate number of pulse phase bins
    unsigned choose_nbin ();

    //! Engine used to perform folding
    class Engine;
    void set_engine (Engine*);

  protected:

    //! The transformation folds the data into the profile
    virtual void transformation ();

    //! Fold nblock blocks of data
    virtual void fold (uint64_t nweights, const unsigned* weights,
		       unsigned ndatperweight, unsigned weight_idat);

    //! Set the idat_start and ndat_fold attributes
    virtual void set_limits (const Observation* input);

    //! Used by the MultiFold class
    void set_idat_start(uint64_t _idat_start){ idat_start = _idat_start; }
    //! Used by the MultiFold class
    uint64_t get_idat_start(){ return idat_start; }
    //! Used by the MultiFold class
    void set_ndat_fold(uint64_t _ndat_fold){ ndat_fold = _ndat_fold; }
    //! Used by the MultiFold class
    uint64_t get_ndat_fold(){ return ndat_fold; }

    //! Called by fold to return pfold
    double get_pfold (const MJD& start_time);
    
    //! Called by new_fold() to return phi
    double get_phi (const MJD& start_time);

    //! Period at which to fold data
    double folding_period;

    //! Set when Tranformation::input is a Weighted TimeSeries
    Reference::To<const WeightedTimeSeries> weighted_input;

    //! Number of phase bins into which the data will be integrated
    unsigned folding_nbin;

    //! Number of phase bins set using set_nbin
    unsigned requested_nbin;

    bool force_sensible_nbin;

    //! Reference phase (phase of bin zero)
    double reference_phase;

    //! Number of polynomial coefficients in model
    unsigned ncoef;

    //! Number of minutes over which polynomial coefficients are valid
    unsigned nspan;

    //! The attributes to be changed in the output PhaseSeries
    Reference::To<const ObservationChange> change;

    //! Flag that the predictor is built for the given ephemeris and input
    bool built;

    //! INTERNAL: the time sample at which to start folding
    uint64_t idat_start;

    //! INTERNAL: the number of time samples to fold
    uint64_t ndat_fold;

    //! Makes sure parameters are initialised
    // This is called from prepare() rather than the constructor so that reducer
    // can set parameters (globally) if they have not been initialised locally for a given
    // dsp::Fold
    void initialise();

  private:

    // Generates folding_predictor from the given ephemeris
    Pulsar::Predictor* get_folding_predictor (const Pulsar::Parameters*,
					      const Observation*);

    //! Phase model with which to fold data (PSR)
    Reference::To<const Pulsar::Predictor> folding_predictor;

    //! Ephemeris with which to create the phase model
    Reference::To<const Pulsar::Parameters> pulsar_ephemeris;

    //! The folding period last used in the fold method
    double pfold;

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

  };

  class Fold::Engine : public Reference::Able
  {
  public:

    void set_parent (Fold*);

    virtual void set_nbin (unsigned nbin) = 0;
    virtual void set_bin (uint64_t idat, unsigned ibin) = 0;

    virtual void setup (uint64_t idat_fold);
    virtual void fold () = 0;

  protected:

    float* output;
    unsigned output_span;

    const float* input;
    unsigned input_span;

    unsigned ndat_fold;

    unsigned nchan, npol, ndim;

    Fold* parent;
  }; 

}

#endif // !defined(__Fold_h)
