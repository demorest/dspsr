//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TwoBitCorrection.h,v $
   $Revision: 1.24 $
   $Date: 2003/06/13 15:02:40 $
   $Author: wvanstra $ */

#ifndef __TwoBitCorrection_h
#define __TwoBitCorrection_h

#include <vector>

#include "dsp/Unpacker.h"

#include "environ.h"

namespace dsp {

  class TwoBitTable;
  class WeightedTimeSeries;

  //! Converts a TimeSeries from two-bit digitized to floating-point values
  /*! The conversion method, poln_unpack, implements the dynamic
    level-setting technique described by Jenet & Anderson (1998, PASP,
    110, 1467; hereafter JA98).  It requires that each byte contains four samples
    from one digitized signal.  If the digitized bits from different convertors
    (ie. different polarizations and/or in-phase and quadrature components are mixed
    within each byte), it is recommended to inherit the SubByteTwoBitCorrection
    class */
  class TwoBitCorrection: public Unpacker {

  public:

    //! Optimal fraction of total power for two-bit sampling threshold
    static const double optimal_threshold;

    //! Maintain a diagnostic histogram of digitizer statistics
    static bool keep_histogram;

    //! Null constructor
    TwoBitCorrection (const char* name = "TwoBitCorrection");

    //! Virtual destructor
    virtual ~TwoBitCorrection ();

    //! Get the number of digitizers
    virtual unsigned get_ndig () const;

    //! Get the number of digitizer outputs in one byte
    virtual unsigned get_ndig_per_byte () const;

    //! Return the offset (number of bytes) into input for the given digitizer
    virtual unsigned get_input_offset (unsigned idig) const;

    //! Return the offset (number of floats) into output for the given digitizer
    virtual unsigned get_output_offset (unsigned idig) const;

    //! Return the output polarization for the given digitizer
    virtual unsigned get_output_ipol (unsigned idig) const;

    //! Return the offset to the next byte containing the current digitizer data
    virtual unsigned get_input_incr () const;

    //! Return the offset (number of floats) between consecutive digitizer samples
    virtual unsigned get_output_incr () const;

    //! Return a descriptive string
    //virtual const string descriptor () const;

    //! Initialize from a descriptor string as output by above
    //virtual void initialize (const string& descriptor);

    //! Set the number of time samples used to estimate undigitized power
    void set_nsample (unsigned nsample);

    //! Get the number of time samples used to estimate undigitized power
    unsigned get_nsample () const { return nsample; }

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
    TwoBitTable* get_table () const { return table; }

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

    //! Get the histogram for the given digitizer
    template <typename T> void get_histogram (vector<T>& data, unsigned idig) const;

    //! Get the centroid of the histogram for the given digitizer
    double get_histogram_mean (unsigned idig) const;

    //! Get the total number of samples in the histogram
    unsigned long get_histogram_total (unsigned idig) const;

    //! Reset histogram counts to zero
    void zero_histogram ();

    //! Return a pointer to a new instance of the appropriate sub-class
    static TwoBitCorrection* create (const BitSeries& input,
				     unsigned nsample=0, float cutoff_sigma=3.0);

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

    //! Number of samples used to estimate undigitized power
    unsigned nsample;

    //! Cut off power for impulsive interference excision
    float cutoff_sigma;

    //! Minumum number of ones in nsample points
    unsigned n_min;

    //! Maximum number of ones in nsample points
    unsigned n_max;

    //! Lookup table and histogram dimensions reflect the attributes
    bool built;

    //! Histograms of number of ones in nsample points
    vector< vector< unsigned long > > histograms;

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

template <typename T> 
void dsp::TwoBitCorrection::get_histogram (vector<T>& data, unsigned idig) const
{
  data.resize (nsample);

  for (unsigned i=0; i<nsample; i++)
    data[i] = T(histograms[idig][i]);
}

#endif
