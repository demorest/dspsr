//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/Fold.h,v $
   $Revision: 1.28 $
   $Date: 2004/06/13 07:40:39 $
   $Author: hknight $ */



#ifndef __Fold_h
#define __Fold_h

#include <vector>

#include "dsp/Observation.h"
#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/PhaseSeries.h"

class polyco;
class psrephem;

namespace dsp {
  class MultiFold;
}

namespace dsp {

  class WeightedTimeSeries;

  //! Fold TimeSeries data into phase-averaged profile(s)
  /*! 
    This Operation does not modify the TimeSeries.  Rather, it accumulates
    the (folded) average pulse profile data within its data structures.
  */
  class Fold : public Transformation <const TimeSeries, PhaseSeries> {

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

    //! Set the period at which to fold data for all sources (in seconds- negative for don't use)
    void set_folding_period (double folding_period);
    //! Set the period at which to fold data, but only do it for this source (in seconds)
    void set_folding_period (double folding_period, string _folding_period_source);

    //! Get the period at which to fold data (in seconds)
    double get_folding_period () const;

    //! Get the phase model which is currently being used to fold data
    const polyco* get_folding_polyco () const;

    //! Get the ephemeris which is currently being used to create the phase model
    const psrephem* get_pulsar_ephemeris () const;

    //! Set the reference phase (phase of bin zero)
    void set_reference_phase (double phase);
    //! Get the reference phase (phase of bin zero)
    double get_reference_phase () const { return reference_phase; }

    //! Set the DM to go in the output PhaseSeries (A negative DM signifies to just use the ephemeris DM)
    void set_dispersion_measure(double _dispersion_measure)
    { dispersion_measure = _dispersion_measure; }

    //! Inquire the DM to go in the output PhaseSeries (A negative DM signifies to just use the ephemeris DM)
    double get_dispersion_measure(){ return dispersion_measure; }

    //! Give the output PhaseSeries the filename its Archive should be written to
    void set_archive_filename(string _archive_filename){ archive_filename = _archive_filename; }

    //! Inquire the filename the Archive will be written to, if any specified
    string get_archive_filename(){ return archive_filename; }

    //! Give the output PhaseSeries the filename extension its Archive will be given
    void set_archive_filename_extension(string _archive_filename_extension)
    { archive_filename_extension = _archive_filename_extension; }

    //! Inquire the filename extension the Archive will be given, if any specified
    string get_archive_filename_extension(){ return archive_filename_extension; }    

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

    //! Sets the 'idat_start' variable based on how much before the folded data ends the input starts
    void workout_idat_start(const Observation* input);

    //! Used by the MultiFold class
    void set_idat_start(uint64 _idat_start){ idat_start = _idat_start; }
    //! Used by the MultiFold class
    uint64 get_idat_start(){ return idat_start; }
    //! Used by the MultiFold class
    void set_ndat_fold(uint64 _ndat_fold){ ndat_fold = _ndat_fold; }
    //! Used by the MultiFold class
    uint64 get_ndat_fold(){ return ndat_fold; }

    //! Period at which to fold data
    double folding_period;

    //! The source name for which to fold at 'folding_period'.  If this is null then all sources are folded at 'folding_period'
    string folding_period_source;

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

    //! Flag that the polyco is built for the given ephemeris and input
    bool built;

    //! The DM to go in the output PhaseSeries
    //! If less than zero, the ephemeris DM is used
    double dispersion_measure;

    //! Used to specify the final output Archive filename
    string archive_filename;

    //! Used to specify the final output Archive filename extension
    string archive_filename_extension;

    //! Polycos from which to choose
    vector< Reference::To<polyco> > polycos;

    //! Ephemerides from which to choose
    vector< Reference::To<psrephem> > ephemerides;

    //! INTERNAL: the time sample at which to start folding
    uint64 idat_start;

    //! INTERNAL: the number of time samples to fold
    uint64 ndat_fold;

  private:

    // Generates folding_polyco from the given ephemeris
    Reference::To<polyco> get_folding_polyco(const psrephem* pephemeris,
					     const Observation* observation);

    //! Set the phase model with which to fold data
    void set_folding_polyco (const polyco* folding_polyco);

    //! Set the ephemeris with which to create the phase model
    void set_pulsar_ephemeris (const psrephem* pulsar_ephemeris);

    //! Phase model with which to fold data (PSR)
    Reference::To<const polyco> folding_polyco;

    //! Ephemeris with which to create the phase model
    Reference::To<const psrephem> pulsar_ephemeris;

    //! The folding period last used in the fold method
    double pfold;

  };
}

#endif // !defined(__Fold_h)





