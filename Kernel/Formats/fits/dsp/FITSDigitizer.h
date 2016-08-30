//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// Recall that PSRFITS search mode data are in TPF order, which complicates
// packing into bytes.

#ifndef __FITSDigitizer_h
#define __FITSDigitizer_h

#include "dsp/Digitizer.h"

namespace dsp
{  
  //! Converts floating point values to N-bit PSRFITS search-mode format
  class FITSDigitizer: public Digitizer
  {
  public:

    //! Default constructor
    FITSDigitizer (unsigned _nbit);

    //! Default destructor
    ~FITSDigitizer ();

    unsigned get_nbit () const {return nbit;}

    //! Set the number of samples to rescale before digitization.
    //! The default is 0, i.e. rescaling must be done elsewhere.
    void set_rescale_samples (unsigned nsamp);

    //! Set the number of blocks to remember when computing scales.
    //! The default is 1, corresponding to no memory.
    void set_rescale_nblock (unsigned nsamp);

    //! If true, leave scales/offsets constant after first measurement.
    void set_rescale_constant (bool rconst);

    //virtual void transformation ();

    //! Pack the data
    void pack ();

    //! Return minimum samples
    // TODO -- is this needed?
    uint64_t get_minimum_samples () { return 2048; }

    void get_scales (std::vector<float>* dat_scl, std::vector<float>* dat_offs);

    Callback<FITSDigitizer*> update;

  protected:

    void set_nbit (unsigned);

    //! rescale input based on mean / variance
    void rescale_pack ();

    void init ();
    void measure_scale ();

    void set_digi_scales();

    //! keep track of first time through scale-measuring procedure
    unsigned rescale_nsamp;
    unsigned rescale_idx;
    unsigned rescale_nblock;
    unsigned rescale_counter;

    //! Keep scaling/offset constant after first estimate.
    bool rescale_constant;

    float digi_mean,digi_scale;
    int digi_min,digi_max;

    //! arrays for accumulating and storing scales
    double *freq_totalsq, *freq_total, *scale, *offset;

  };
}

#endif
