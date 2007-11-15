//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/PhaseSeries.h,v $
   $Revision: 1.24 $
   $Date: 2007/11/15 10:42:19 $
   $Author: straten $ */

#ifndef __PhaseSeries_h
#define __PhaseSeries_h

#include "dsp/TimeSeries.h"

namespace Pulsar {
  class Predictor;
  class Parameters;
}

namespace dsp {
  
  class PhaseSeries : public TimeSeries {

    friend class Fold;

  public:

    //! Default constructor
    PhaseSeries ();

    //! Copy constructor
    PhaseSeries (const PhaseSeries&);

    //! Assigment operator
    PhaseSeries& operator = (const PhaseSeries&);

    //! Destructor
    ~PhaseSeries ();

    //! Clone operator
    PhaseSeries* clone() const;

    //! Allocate the space required to store nsamples time samples.
    virtual void resize (int64 nsamples);

    //! Add prof to this
    PhaseSeries& operator += (const PhaseSeries& prof);

    //! Set the reference phase (phase of bin zero)
    void set_reference_phase (double phase) { reference_phase = phase; }
    //! Get the reference phase (phase of bin zero)
    double get_reference_phase () const { return reference_phase; }

    //! Set the period at which to fold data (in seconds)
    /*! The Pulsar::Predictor and ephemeris are set to null values upon
      setting the folding period */
    void set_folding_period (double _folding_period);
    //! Get the period at which to fold data (in seconds)
    double get_folding_period () const;

    //! Set the phase predictor with which the data were folded 
    void set_folding_predictor (const Pulsar::Predictor*);

    //! Get the phase predictor with which the data were folded
    const Pulsar::Predictor* get_folding_predictor () const;

    //! Set the pulsar ephemeris used to fold.
    void set_pulsar_ephemeris (const Pulsar::Parameters*);
    
    //! Returns the pulsar ephemeris stored
    const Pulsar::Parameters* get_pulsar_ephemeris() const;

    //! Get the number of seconds integrated
    double get_integration_length () const { return integration_length; }

    //! Increment the integration length
    void increment_integration_length (double seconds)
    { integration_length += seconds; }

    //! Get the end time
    MJD get_end_time () const { return end_time; }

    //! Set the end time
    void set_end_time (const MJD& mjd) { end_time = mjd; }

    //! Get the number of phase bins
    unsigned get_nbin () const { return unsigned(get_ndat()); }

    //! Get the hit for the given bin
    unsigned get_hit (unsigned ibin) const { return hits[ibin]; }

    //! Get the hits array
    unsigned* get_hits () { return &hits[0]; }

    //! Set the hits in all bins
    void set_hits (unsigned value);

    //! Get the mid-time of the integration
    MJD get_mid_time () const;

    //! Reset all phase bin totals to zero
    void zero ();

    //! Over-ride Observation::combinable_rate
    bool combinable_rate (double) const { return true; }

#if 0
    Wvs FIX LATER ?
    //! Store what the output Archive's filename should be
    void set_archive_filename(std::string _archive_filename)
    { archive_filename = _archive_filename; }

    //! Inquire what the output Archive's filename is going to be (if anything)
    string get_archive_filename() const { return archive_filename; }

    //! Store what the output Archive's filename extension should be
    void set_archive_filename_extension(std::string _archive_filename_extension)
    { archive_filename_extension = _archive_filename_extension; }

    //! Inquire what the output Archive's filename extension is going to be (if anything)
    string get_archive_filename_extension() const { return archive_filename_extension; }

#endif

  protected:

    //! Period at which CAL data is folded
    double folding_period;

    //! Phase polynomial(s) with which PSR is folded
    Reference::To<const Pulsar::Predictor> folding_predictor;

    //! The ephemeris (if any) that was used to generate the Pulsar::Predictor
    Reference::To<const Pulsar::Parameters> pulsar_ephemeris;

    //! Reference phase (phase of bin zero)
    double reference_phase;

    //! Number of time samples integrated into each phase bin
    std::vector<unsigned> hits;

    //! The number of seconds integrated into the profile(s)
    double integration_length;

    //! The MJD of the last-integrated time sample's tail edge
    MJD end_time;

    //! Return true when Observation can be integrated (and prepare for it)
    bool mixable (const Observation& obs, unsigned nbin,
		  int64 istart=0, int64 fold_ndat=0);

#if 0
    WvS FIX LATER ?
    //! The Archive::unload_filename attribute
    string archive_filename;

    //! This filename extension will be added onto the Archive::unload_filename attribute
    string archive_filename_extension;
#endif

  };

}

#endif
