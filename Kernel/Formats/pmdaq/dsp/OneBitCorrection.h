//-*-C++-*-

#ifndef __OneBitCorrection_h
#define __OneBitCorrection_h

class OneBitCorrection;

#include <vector>

#include "dsp/Unpacker.h"
#include "dsp/Timeseries.h"
#include "dsp/Bitseries.h"

#include "environ.h"

namespace dsp {

  //! Converts a Timeseries from one-bit digitized to floating-point values
  class OneBitCorrection: public Unpacker {

  public:

    //! Maintain a diagnostic histogram of digitizer statistics
    static bool keep_histogram;

    //! Null constructor
    OneBitCorrection (const char* name = "OneBitCorrection");

    //! Virtual destructor
    virtual ~OneBitCorrection ();

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

    //
    //
    //

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
    static OneBitCorrection* create (const Bitseries& input,
				     int nsample=0, float cutoff_sigma=3.0);

    //! Return the high and low output voltage values
    static void output_levels (float p_in, float& lo, float& hi, float& A);

  protected:

    //! Perform the bit conversion operation on the input Timeseries
    virtual void operation ();

    //! Unpacking algorithm may be re-defined by sub-classes
    virtual void unpack ();

    //! Return true if OneBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);


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
void dsp::OneBitCorrection::get_histogram (vector<T>& data, int chan) const
{
  data.resize (nsample);

  for (int i=0; i<nsample; i++)
    data[i] = T(histograms[chan][i]);
}

#endif
