//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TwoBitCorrection.h,v $
   $Revision: 1.5 $
   $Date: 2002/07/10 06:59:34 $
   $Author: wvanstra $ */

#ifndef __TwoBitCorrection_h
#define __TwoBitCorrection_h

#include <vector>

#include "Operation.h"
#include "environ.h"

namespace dsp {
  
  //! Converts a Timeseries from 2-bit digitized to floating point values
  /*! The conversion routines are implemented by the TwoBitCorrection
    sub-classes, which perform the dynamic level-setting technique
    described by Jenet & Anderson (1998, PASP, 110, 1467; hereafter
    JA98) in order to correct for the effects of digitization */
  class TwoBitCorrection: public Operation {

  public:

    //! Optimal fraction of total power for 2-bit sampling threshold
    static const double optimal_threshold;

    //! Maintain a diagnostic histogram of digitizer statistics
    static bool keep_histogram;

    //! Null constructor
    TwoBitCorrection (const char* name = "TwoBitCorrection",
		   Behaviour type = outofplace);

    //! Virtual destructor
    virtual ~TwoBitCorrection () { destroy(); }

    //! Return a descriptive string
    virtual const string descriptor () const;

    //! Initialize from a descriptor string as output by above
    virtual void initialize (const string& descriptor);

    //! Build the dynamic level setting lookup table
    virtual void build (int nchannel, int nsample, float cutoff_sigma);

    //! Get the low and hi output levels
    virtual void get_output_levels (int nlo, float& lo, float& hi);

    //! Return true if val corresponds to a digitized high voltage state
    virtual bool get_hi (unsigned char val);

    //! Return the sign of the digitized voltage state
    virtual int get_sign (unsigned char val);

    //! Calculate the sum and sum-squared from each channel of digitized data
    virtual int64 stats (vector<double>& sum, vector<double>& sumsq);

    //! Get the number of digitizer channels
    int get_nchannel () const { return nchannel; }

    //! Get the number of samples used to estimate undigitized power
    int get_nsample () const { return nsample; }

    //! Get the cut off power for impulsive interference excision
    float get_cutoff_sigma() const { return cutoff_sigma; }

    //! Get the minumum number of ones in nsample points
    int get_nmin() const { return n_min; }

    //! Get the maxumum number of ones in nsample points
    int get_nmax() const { return n_max; }

    //! Reset histogram counts to zero
    void zero_histogram ();

    //! Get the centroid of the histogram for the given digitizer channel
    double get_histogram_mean (int channel);

    //! Return a pointer to a new instance of the appropriate sub-class
    static TwoBitCorrection* create (const Timeseries& input,
				     int nsample=0, float cutoff_sigma=3.0);

  protected:

    //! Perform the bit conversion operation on the input Timeseries
    virtual void operation ();

    //! Unpacking algorithm is defined by sub-classes
    virtual void unpack () = 0;

    //! Number of digitizer channels
    int nchannel;

    //! Histograms of number of ones in nsample points
    unsigned long* histograms;

    //! Values used in Dynamic Level Setting
    float* dls_lookup;

    //! Number of samples used to estimate undigitized power
    int nsample;

    //! Cut off power for impulsive interference excision
    float cutoff_sigma;

    //! Minumum number of ones in nsample points
    int n_min;

    //! Maximum number of ones in nsample points
    int n_max;

    //! Set limits using current attributes
    void set_twobit_limits ();

    //! Destroy allocated resources
    void destroy ();

    //! Allocate resources
    void allocate ();
  };
  
}

#endif
