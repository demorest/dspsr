//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TwoBitCorrection.h,v $
   $Revision: 1.13 $
   $Date: 2002/10/07 02:13:22 $
   $Author: wvanstra $ */

#ifndef __TwoBitCorrection_h
#define __TwoBitCorrection_h

#include <vector>

#include "Operation.h"
#include "environ.h"

namespace dsp {

  class TwoBitTable;
  
  //! Converts a Timeseries from two-bit digitized to floating-point values
  /*! The conversion routines are implemented by the TwoBitCorrection
    sub-classes, which perform the dynamic level-setting technique
    described by Jenet & Anderson (1998, PASP, 110, 1467; hereafter
    JA98) in order to correct for the effects of digitization */
  class TwoBitCorrection: public Operation {

  public:

    //! Optimal fraction of total power for two-bit sampling threshold
    static const double optimal_threshold;

    //! Maintain a diagnostic histogram of digitizer statistics
    static bool keep_histogram;

    //! Null constructor
    TwoBitCorrection (const char* name = "TwoBitCorrection",
		      Behaviour type = outofplace);

    //! Virtual destructor
    virtual ~TwoBitCorrection ();

    //! Return a descriptive string
    //virtual const string descriptor () const;

    //! Initialize from a descriptor string as output by above
    //virtual void initialize (const string& descriptor);

    //! Set the number of time samples used to estimate undigitized power
    void set_nsample (int nsample);

    //! Get the number of time samples used to estimate undigitized power
    int get_nsample () const { return nsample; }

    //! Set the cut off power for impulsive interference excision
    void set_cutoff_sigma (float cutoff_sigma);
    
    //! Get the cut off power for impulsive interference excision
    float get_cutoff_sigma() const { return cutoff_sigma; }

    //! Get the number of digitizer channels
    int get_nchannel () const { return nchannel; }

    //! Set the digitization convention
    void set_table (TwoBitTable* table);

    //! Get the digitization convention
    TwoBitTable* get_table () const { return table; }

    //
    //
    //

    //! Get the optimal fraction of low voltage levels
    virtual float get_optimal_fraction_low () const;

    //! Calculate the sum and sum-squared from each channel of digitized data
    virtual int64 stats (vector<double>& sum, vector<double>& sumsq);

    //! Get the minumum number of ones in nsample points
    int get_nmin() const { return n_min; }

    //! Get the maxumum number of ones in nsample points
    int get_nmax() const { return n_max; }

    //! Get the histogram for the given channel
    template <typename T> void get_histogram (vector<T>& data, int chan) const;

    //! Get the centroid of the histogram for the given digitizer channel
    double get_histogram_mean (int channel) const;

    //! Get the total number of samples in the histogram
    unsigned long get_histogram_total (int channel) const;

    //! Reset histogram counts to zero
    void zero_histogram ();

    //! Return a pointer to a new instance of the appropriate sub-class
    static TwoBitCorrection* create (const Timeseries& input,
				     int nsample=0, float cutoff_sigma=3.0);

    //! Return the high and low output voltage values
    static void output_levels (float p_in, float& lo, float& hi, float& A);

    //! Generate dynamic level setting and scattered power correction lookup
    static void generate (float* dls, float* spc,
			  int n_min, int n_max, int n_tot,
			  TwoBitTable* table, bool huge);

  protected:

    //! Perform the bit conversion operation on the input Timeseries
    virtual void operation ();

    //! Build the two-bit correction look-up table and allocate histograms
    virtual void build ();

    //! Unpacking algorithm may be re-defined by sub-classes
    virtual void unpack ();

    //! Unpack a single polarization from raw into data
    virtual void poln_unpack (float* data, const unsigned char* raw, 
			      uint64 ndat, unsigned long* hist, unsigned gap);

    //! Two-bit conversion table generator
    Reference::To<TwoBitTable> table;

    //! Number of digitizer channels
    int nchannel;

    //! Number of digitizer channels packed into each byte
    int channels_per_byte;

    //! Number of samples used to estimate undigitized power
    int nsample;

    //! Cut off power for impulsive interference excision
    float cutoff_sigma;

    //! Minumum number of ones in nsample points
    int n_min;

    //! Maximum number of ones in nsample points
    int n_max;

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

  };
  
}

template <typename T> 
void dsp::TwoBitCorrection::get_histogram (vector<T>& data, int chan) const
{
  data.resize (nsample);

  for (int i=0; i<nsample; i++)
    data[i] = T(histograms[chan][i]);
}

#endif
