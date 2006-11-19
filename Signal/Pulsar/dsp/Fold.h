//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/Fold.h,v $
   $Revision: 1.46 $
   $Date: 2006/11/19 15:39:25 $
   $Author: straten $ */

#ifndef __baseband_dsp_Fold_h
#define __baseband_dsp_Fold_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/PhaseSeries.h"

class polyco;
class psrephem;
class rawprofile;

namespace dsp {

  class WeightedTimeSeries;

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
    Fold (bool _dont_set_limits = false);
    
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

    //! Set the name of the source
    void set_source_name (const std::string& name);
    //! Get the name of the source
    std::string get_source_name () const;

    //! Set the period at which to fold data for all sources (in seconds- negative for don't use)
    void set_folding_period (double folding_period);
    //! Set the period at which to fold data, but only do it for this source (in seconds)
    void set_folding_period (double folding_period, std::string _folding_period_source);

    //! Get the period at which data is being folded at (in seconds)
    double get_folding_period () const;

    //! Returns true if data will be folded at a constant period
    bool has_folding_period() const { return (folding_period>0); }

    //! Get the phase model which is currently being used to fold data
    const polyco* get_folding_polyco () const;

    //! Returns true if data will be folded using polyco
    bool has_folding_polyco () const;

    //! Get the ephemeris used to create the phase model
    //   const psrephem* get_pulsar_ephemeris () const;

    //! Get the ephemeris used to create the phase model
    const psrephem* get_pulsar_ephemeris () const;

    //! use to see whether pulsar eph is stored
    bool has_pulsar_ephemeris() const { return pulsar_ephemeris.ptr(); }

    //! Set the ephemeris with which to create a new phase model
    void set_pulsar_ephemeris (const psrephem* pulsar_ephemeris);

    //! Set the reference phase (phase of bin zero)
    void set_reference_phase (double phase);
    //! Get the reference phase (phase of bin zero)
    double get_reference_phase () const { return reference_phase; }

#if 0

    Wvs FIX LATER ?
    //! Give the output PhaseSeries the filename its Archive should be written to
    void set_archive_filename(string _archive_filename){ archive_filename = _archive_filename; }

    //! Inquire the filename the Archive will be written to, if any specified
    string get_archive_filename(){ return archive_filename; }

    //! Give the output PhaseSeries the filename extension its Archive will be given
    void set_archive_filename_extension(string _archive_filename_extension)
    { archive_filename_extension = _archive_filename_extension; }

    //! Inquire the filename extension the Archive will be given, if any specified
    string get_archive_filename_extension(){ return archive_filename_extension; }    

#endif

    //! Overload Transformation::set_input to set weighted_input
    void set_input (const TimeSeries* input);

    //! Add a phase model with which to choose to fold the data
    void add_folding_polyco (const polyco* folding_polyco);

    //! Add an ephemeris with which to choose to create the phase model
    void add_pulsar_ephemeris (const psrephem* pulsar_ephemeris);

    //! Choose an appropriate ephemeris from those added
    const psrephem* choose_ephemeris (const std::string& pulsar);

    //! Choose an appropriate polyco from those added
    const polyco* choose_polyco (const MJD& time, const std::string& pulsar);

    //! Choose an appropriate number of pulse phase bins
    unsigned choose_nbin ();

    //! Fold nblock blocks of data
    //! This is kept in for use in baseband/timeseries/rawprofile.C
    //! new_fold() is now called which doesn't assume chanpols are contiguous and uses get_datptr()
    void fold (double& integration_length, float* phase, unsigned* hits,
	       const Observation* info,unsigned nblock,
	       const float* time, uint64 ndat, unsigned ndim,
	       const unsigned* weights=0, unsigned ndatperweight=0,
	       uint64 idat_start=0, uint64 ndat_fold=0);

    void set_dispersion_measure(float _dispersion_measure)
    { dispersion_measure = _dispersion_measure; }
    float get_dispersion_measure() const
    { return dispersion_measure; }

  protected:

    //! The transformation folds the data into the profile
    virtual void transformation ();

    //! Fold nblock blocks of data
    virtual void fold (uint64 nweights, const unsigned* weights,
		       unsigned ndatperweight, unsigned weight_idat);

    //! Set the idat_start and ndat_fold attributes
    virtual void set_limits (const Observation* input);

    //! Used by the MultiFold class
    void set_idat_start(uint64 _idat_start){ idat_start = _idat_start; }
    //! Used by the MultiFold class
    uint64 get_idat_start(){ return idat_start; }
    //! Used by the MultiFold class
    void set_ndat_fold(uint64 _ndat_fold){ ndat_fold = _ndat_fold; }
    //! Used by the MultiFold class
    uint64 get_ndat_fold(){ return ndat_fold; }

    //! Called by fold to return pfold
    double get_pfold (const MJD& start_time);
    
    //! Called by new_fold() to return phi
    double get_phi (const MJD& start_time);

    //! Period at which to fold data
    double folding_period;

    //! The source name for which to fold at folding_period.
    /*! If this is not set, then all sources are folded at folding_period */
    std::string folding_period_source;

    //! Set when Tranformation::input is a Weighted TimeSeries
    Reference::To<const WeightedTimeSeries> weighted_input;

    //! Number of phase bins into which the data will be integrated
    unsigned folding_nbin;

    //! Number of phase bins set using set_nbin
    unsigned requested_nbin;

    //! Reference phase (phase of bin zero)
    double reference_phase;

    //! Number of polynomial coefficients in model
    unsigned ncoef;

    //! Number of minutes over which polynomial coefficients are valid
    unsigned nspan;

    //! The name of the source (overrides input source name)
    std::string source_name;

    //! Flag that the polyco is built for the given ephemeris and input
    bool built;

#if 0
    WvS FIX LATER ?
    //! Used to specify the final output Archive filename
    string archive_filename;

    //! Used to specify the final output Archive filename extension
    string archive_filename_extension;
#endif

    //! Polycos from which to choose
    std::vector< Reference::To<const polyco> > polycos;

    //! Ephemerides from which to choose
    std::vector< Reference::To<const psrephem> > ephemerides;

    //! INTERNAL: the time sample at which to start folding
    uint64 idat_start;

    //! INTERNAL: the number of time samples to fold
    uint64 ndat_fold;

    //! Makes sure parameters are initialised
    // This is called from prepare() rather than the constructor so that reducer
    // can set parameters (globally) if they have not been initialised locally for a given
    // dsp::Fold
    void initialise();

    //! Set the phase model with which to fold data
    void set_folding_polyco (const polyco* folding_polyco);

  private:

    // Generates folding_polyco from the given ephemeris
    Reference::To<polyco> get_folding_polyco(const psrephem* pephemeris,
					     const Observation* observation);

    //! Phase model with which to fold data (PSR)
    Reference::To<const polyco> folding_polyco;

    //! Ephemeris with which to create the phase model
    Reference::To<const psrephem> pulsar_ephemeris;

    //! The folding period last used in the fold method
    double pfold;

    //! If true, set_limits() returns straight away
    bool dont_set_limits;

    //! Dispersion measure to install in output/ephemeris
    double dispersion_measure;

  };
}

#endif // !defined(__Fold_h)





