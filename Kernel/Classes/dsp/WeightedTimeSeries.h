//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/WeightedTimeSeries.h,v $
   $Revision: 1.2 $
   $Date: 2002/11/11 12:19:32 $
   $Author: wvanstra $ */

#ifndef __WeightedTimeSeries_h
#define __WeightedTimeSeries_h

#include "dsp/TimeSeries.h"

namespace dsp {
  
  //! Container of weighted time-major order floating point data.
  /* The WeightedTimeSeries class contains floating point data that
     may be flagged as bad in the time domain. */
  class WeightedTimeSeries : public TimeSeries {

  public:
    //! Null constructor
    WeightedTimeSeries ();

    //! Set this equal to copy
    virtual WeightedTimeSeries& operator = (const WeightedTimeSeries& copy);

    //! Add each value in data to this
    virtual WeightedTimeSeries& operator += (const WeightedTimeSeries& data);

    //! Set the number of time samples per weight
    /*! Set ndat_per_weight to zero to effect no weighting of data */
    virtual void set_ndat_per_weight (unsigned ndat_per_weight);

    //! Get the number of time samples per weight
    unsigned get_ndat_per_weight () const { return ndat_per_weight; }

    //! Set the number of polarizations with independent weights
    virtual void set_npol_weight (unsigned npol_weight);

    //! Get the number of polarizations with independent weights
    unsigned get_npol_weight () const { return npol_weight; }

    //! Set the number of frequency channels with independent weights
    virtual void set_nchan_weight (unsigned nchan_weight);

    //! Get the number of frequency channels with independent weights
    unsigned get_nchan_weight () const { return nchan_weight; }

    //! Allocate the space required to store nsamples time samples.
    virtual void resize (uint64 nsamples);
    
    //! Set all values to zero
    virtual void zero ();

    //! For each zero weight, sets all weights to zero
    virtual void mask_weights ();

    //! Check that each floating point value is zeroed if weight is zero
    virtual void check_weights ();

    //! Set all weights to one
    virtual void neutral_weights ();

    //! Get the number of weights
    uint64 get_nweights () const;

    //! Get the weights array for the specfied polarization and frequency
    unsigned* get_weights (unsigned ichan=0, unsigned ipol=0);

    //! Get the weights array for the specfied polarization and frequency
    const unsigned* get_weights (unsigned ichan=0, unsigned ipol=0) const;

  protected:

    void scrunch_weight_check (unsigned nscrunch);
    void convolve_weights (int nfft, int nkeep);

    //! Number of polarizations with independent weights
    unsigned npol_weight;

    //! Number of frequency channels with independent weights
    unsigned nchan_weight;

    //! The number of time samples per weight
    unsigned ndat_per_weight;

  private:
    //! The weights buffer
    unsigned* weights;

    //! The size of the buffer
    uint64 weight_size;

    //! The size of each division of the buffer
    uint64 weight_subsize;
  };

  // zaps data with zeroes
  void convolve_tweights (unsigned* weights,
			  unsigned long nwts,
			  unsigned testwts, unsigned restwts);
  
  // returns the number of weights left in the array
  unsigned long scrunch_tweights (unsigned* weights,
				  unsigned long nwts,
				  unsigned scrunch_factor);
  
}

#endif

