//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/PhaseSeries.h,v $
   $Revision: 1.13 $
   $Date: 2005/03/29 11:48:22 $
   $Author: wvanstra $ */

#ifndef __PhaseSeries_h
#define __PhaseSeries_h

#include <vector>

#include "Reference.h"

#include "dsp/TimeSeries.h"

class psrephem;
class polyco;

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

    //! Allocate the space required to store nsamples time samples.
    virtual void resize (int64 nsamples);

    //! Add prof to this
    PhaseSeries& operator += (const PhaseSeries& prof);

    //! Set the reference phase (phase of bin zero)
    void set_reference_phase (double phase) { reference_phase = phase; }
    //! Get the reference phase (phase of bin zero)
    double get_reference_phase () const { return reference_phase; }

    //! Set the period at which to fold data (in seconds)
    //! The polyco and ephemeris are set to null values upon setting of the folding period
    void set_folding_period (double _folding_period);
    //! Get the period at which to fold data (in seconds)
    double get_folding_period () const;

    //! Get the phase polynomial(s) with which to fold data
    const polyco* get_folding_polyco () const;

    //! Set the pulsar ephemeris used to fold with.  User must also supply the polyco that was generated from the ephemeris and used for folding
    void set_pulsar_ephemeris(const psrephem* _pulsar_ephemeris, const polyco* _folding_polyco);
    
    //! Returns the pulsar ephemeris stored
    const psrephem* get_pulsar_ephemeris() const;

    //! Get the number of seconds integrated
    double get_integration_length () const { return integration_length; }

    virtual MJD get_end_time () const { return end_time; }

    //! Get the number of phase bins
    unsigned get_nbin () const { return get_ndat(); }

    //! Get the hit for the given bin
    unsigned get_hit (unsigned ibin) const { return hits[ibin]; }

    //! Get the mid-time of the integration
    MJD get_mid_time () const;

    //! Reset all phase bin totals to zero
    void zero ();

    //! Over-ride Observation::combinable_rate
    bool combinable_rate (double test_rate) const { return true; }

    //! Store what the output Archive's filename should be
    void set_archive_filename(string _archive_filename)
    { archive_filename = _archive_filename; }

    //! Inquire what the output Archive's filename is going to be (if anything)
    string get_archive_filename() const { return archive_filename; }

    //! Store what the output Archive's filename extension should be
    void set_archive_filename_extension(string _archive_filename_extension)
    { archive_filename_extension = _archive_filename_extension; }

    //! Inquire what the output Archive's filename extension is going to be (if anything)
    string get_archive_filename_extension() const { return archive_filename_extension; }

  protected:

    //! Period at which CAL data is folded
    double folding_period;

    //! Phase polynomial(s) with which PSR is folded
    Reference::To<const polyco> folding_polyco;

    //! The ephemeris (if any) that was used to generate the polyco
    Reference::To<const psrephem> pulsar_ephemeris;

    //! Reference phase (phase of bin zero)
    double reference_phase;

    //! Number of time samples integrated into each phase bin
    vector<unsigned> hits;

    //! The number of seconds integrated into the profile(s)
    double integration_length;

    //! The MJD of the last-integrated time sample's tail edge
    MJD end_time;

    //! Return true when Observation can be integrated (and prepare for it)
    bool mixable (const Observation& obs, unsigned nbin,
		  int64 istart=0, int64 fold_ndat=0);

    //! The Archive::unload_filename attribute
    string archive_filename;

    //! This filename extension will be added onto the Archive::unload_filename attribute
    string archive_filename_extension;

  };

}

#endif
