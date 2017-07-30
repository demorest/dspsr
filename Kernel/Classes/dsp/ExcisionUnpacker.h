//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/ExcisionUnpacker.h

#ifndef __ExcisionUnpacker_h
#define __ExcisionUnpacker_h

#include "dsp/HistUnpacker.h"
#include "JenetAnderson98.h"

namespace dsp
{
  class WeightedTimeSeries;
  class Input;

  //! Excises digitized data with statistics outside of acceptable limits
  class ExcisionUnpacker: public HistUnpacker
  {

  public:

    //! Null constructor
    ExcisionUnpacker (const char* name = "ExcisionUnpacker");

    //! Initialize the WeightedTimeSeries dimensions
    void reserve ();

    //! Overload Transformation::set_output to also set weighted_output
    void set_output (TimeSeries*);

    //! Match the ndat_per_weight to the resolution of the Input
    void match_resolution (const Input*);

    //! Return ndat_per_weight
    unsigned get_resolution () const;

    //! Get the offset (number of bytes) into input for the given digitizer
    virtual unsigned get_input_offset (unsigned idig) const;

    //! Get the offset to the next byte containing the current digitizer data
    virtual unsigned get_input_incr () const;

    //! Get the offset (number of floats) between consecutive digitizer samples
    virtual unsigned get_output_incr () const;

    //! Set the number of samples per weight in WeightTimeSeries output
    virtual void set_ndat_per_weight (unsigned ndat_per_weight);

    //! Get the number of samples per weight in WeightTimeSeries output
    unsigned get_ndat_per_weight () const { return ndat_per_weight; }

    //! Set the number of states in the histogram
    virtual void set_nstate (unsigned nstate) { set_ndat_per_weight (nstate); }

    //! Set the cut off power for impulsive interference excision
    virtual void set_cutoff_sigma (float cutoff_sigma);
    
    //! Get the cut off power for impulsive interference excision
    virtual float get_cutoff_sigma() const { return cutoff_sigma; }

    //! Set the sampling threshold as a fraction of the noise power
    virtual void set_threshold (float threshold) { }

    //! Get the sampling threshold as a fraction of the noise power
    virtual float get_threshold () const { return 0.0; }

    //
    //
    //

    //! Get the fraction of samples in the low state
    float get_fraction_low () const { return ja98.get_mean_Phi(); }

    //! Get the minumum number of ones in ndat_per_weight points
    unsigned get_nlow_min() const { return nlow_min; }

    //! Get the maxumum number of ones in ndat_per_weight points
    unsigned get_nlow_max() const { return nlow_max; }

  protected:

    //! Set when Transformation::output is a WeightedTimeSeries
    Reference::To<WeightedTimeSeries> weighted_output;

    //! Set nlow_min and nlow_max using current attributes
    void set_limits ();

    //! Build the look-up tables and allocate histograms
    virtual void build ();

    //! Unpack using dig_unpack
    void unpack ();

    //! Unpack a single digitized stream from raw into data
    virtual void dig_unpack (const unsigned char* input_data, 
			     float* output_data,
			     uint64_t ndat,
			     unsigned long* hist,
			     unsigned* weights = 0,
			     unsigned nweights = 0) = 0;

    //! Template method can be used to implement pure virtual dig_unpack
    template<class U, class Iterator>
    void excision_unpack (U& unpack, Iterator& input,
		          float* output_data, uint64_t ndat,
		          unsigned long* hist,
		          unsigned* weights, unsigned nweights);

    //! Cut off power for impulsive interference excision
    float cutoff_sigma;

    //! Minumum number of low states in ndat_per_weight points
    unsigned nlow_min;

    //! Maximum number of low states in ndat_per_weight points
    unsigned nlow_max;

    //! The theory behind the implementation
    JenetAnderson98 ja98;

    //! The current digitized stream
    unsigned current_digitizer;

    //! Derived types may not set built flag to true, but can set it to false
    void not_built ();

  private:

    //! Lookup table and histogram dimensions reflect the attributes
    bool built;

    //! Number of samples per weight
    unsigned ndat_per_weight;

  };
}

#endif
