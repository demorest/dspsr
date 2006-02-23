//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TwoBitCorrection.h,v $
   $Revision: 1.28 $
   $Date: 2006/02/23 17:52:04 $
   $Author: wvanstra $ */

#ifndef __TwoBitCorrection_h
#define __TwoBitCorrection_h

#include <vector>

#include "dsp/HistUnpacker.h"

#include "environ.h"

namespace dsp {

  class TwoBitTable;
  class WeightedTimeSeries;

  //! Converts BitSeries data from two-bit digitized to floating-point values
  /*! The conversion method, poln_unpack, implements the dynamic
    level-setting technique described by Jenet & Anderson (1998, PASP,
    110, 1467; hereafter JA98).  It requires that each byte contains
    four samples from one digitized signal.  If the digitized bits
    from different convertors (ie. different polarizations and/or
    in-phase and quadrature components) are mixed within each byte, it
    is recommended to inherit the SubByteTwoBitCorrection class. */
  class TwoBitCorrection: public HistUnpacker {

  public:

    //! Optimal fraction of total power for two-bit sampling threshold
    static const double optimal_threshold;

    //! Perform the Jenet and Anderson correction
    static bool change_levels;

    //! Null constructor
    TwoBitCorrection (const char* name = "TwoBitCorrection");

    //! Virtual destructor
    virtual ~TwoBitCorrection ();

    //! Get the number of digitizer outputs in one byte
    virtual unsigned get_ndig_per_byte () const;

    //! Get the offset (number of bytes) into input for the given digitizer
    virtual unsigned get_input_offset (unsigned idig) const;

    //! Get the offset (number of floats) into output for the given digitizer
    virtual unsigned get_output_offset (unsigned idig) const;

    //! Get the output polarization for the given digitizer
    virtual unsigned get_output_ipol (unsigned idig) const;

    //! Get the output frequency channel for the given digitizer;
    virtual unsigned get_output_ichan (unsigned idig) const;

    //! Get the offset to the next byte containing the current digitizer data
    virtual unsigned get_input_incr () const;

    //! Get the offset (number of floats) between consecutive digitizer samples
    virtual unsigned get_output_incr () const;

    //! Return a descriptive string
    //virtual const string descriptor () const;

    //! Initialize from a descriptor string as output by above
    //virtual void initialize (const string& descriptor);

    //! Set the number of time samples used to estimate undigitized power
    void set_nsample (unsigned nsample);

    //! Set the sampling threshold as a fraction of the noise power
    void set_threshold (float threshold);

    //! Get the sampling threshold as a fraction of the noise power
    float get_threshold () const { return threshold; }

    //! Set the cut off power for impulsive interference excision
    void set_cutoff_sigma (float cutoff_sigma);
    
    //! Get the cut off power for impulsive interference excision
    float get_cutoff_sigma() const { return cutoff_sigma; }

    //! Set the digitization convention
    void set_table (TwoBitTable* table);

    //! Get the digitization convention
    const TwoBitTable* get_table () const;

    //! Overload Transformation::set_output to set weighted_output
    void set_output (TimeSeries* output);

    //
    //
    //

    //! Get the fraction of low voltage levels
    virtual float get_fraction_low () const;

    //! Calculate the sum and sum-squared from each digitizer
    virtual int64 stats (vector<double>& sum, vector<double>& sumsq);

    //! Get the minumum number of ones in nsample points
    unsigned get_nmin() const { return n_min; }

    //! Get the maxumum number of ones in nsample points
    unsigned get_nmax() const { return n_max; }

    //! Return a pointer to a new instance of the appropriate sub-class
    static TwoBitCorrection* create (const BitSeries& input,
				     unsigned nsample=0, float cutoff_rms=3.0);

    //! Return the high and low output voltage values
    static void output_levels (float p_in, float& lo, float& hi, float& A);

    //! Generate dynamic level setting and scattered power correction lookup
    static void generate (float* dls, float* spc,
			  unsigned n_min, unsigned n_max, unsigned n_tot,
			  TwoBitTable* table, bool huge);

  protected:

    //! Perform the bit conversion transformation on the input TimeSeries
    virtual void transformation ();

    //! Build the two-bit correction look-up table and allocate histograms
    virtual void build ();

    //! Unpacking algorithm may be re-defined by sub-classes
    virtual void unpack ();

    //! Unpack a single polarization from raw into data
    virtual void dig_unpack (float* output_data,
			     const unsigned char* input_data, 
			     uint64 ndat,
			     unsigned digitizer,
			     unsigned* weights = 0,
			     unsigned nweights = 0);

    //! Two-bit conversion table generator
    Reference::To<TwoBitTable> table;

    //! Set when Transformation::output is a WeightedTimeSeries
    Reference::To<WeightedTimeSeries> weighted_output;

    //! Sampling threshold as a fraction of the noise power
    float threshold;

    //! Cut off power for impulsive interference excision
    float cutoff_sigma;

    //! Minumum number of ones in nsample points
    unsigned n_min;

    //! Maximum number of ones in nsample points
    unsigned n_max;

    //! Lookup table and histogram dimensions reflect the attributes
    bool built;

    //! Values used in Dynamic Level Setting
    vector< float > dls_lookup;

    //! Number of low-voltage states in a given byte
    vector< unsigned char > nlo_lookup;

    //! Set limits using current attributes
    void set_limits ();

    //! Build the number of low-voltage states lookup table
    virtual void nlo_build ();

  };
  
}

#endif
