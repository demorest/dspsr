//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/TwoBitCorrection.h,v $
   $Revision: 1.41 $
   $Date: 2008/07/09 04:26:17 $
   $Author: straten $ */

#ifndef __TwoBitCorrection_h
#define __TwoBitCorrection_h

#include "dsp/ExcisionUnpacker.h"

#include "environ.h"
#include <vector>

namespace dsp {

  class TwoBitTable;

  //! Converts BitSeries data from two-bit digitized to floating-point values
  /*! The conversion method, poln_unpack, implements the dynamic output
    level-setting technique described by Jenet & Anderson (1998, PASP,
    110, 1467; hereafter JA98).  It requires that each byte contains
    four samples from one digitized signal.  If the digitized bits
    from different convertors (ie. different polarizations and/or
    in-phase and quadrature components) are mixed within each byte, it
    is recommended to inherit the SubByteTwoBitCorrection class. */
  class TwoBitCorrection: public ExcisionUnpacker
  {

  public:

    //! Perform the Jenet and Anderson correction
    static bool change_levels;

    //! Null constructor
    TwoBitCorrection (const char* name = "TwoBitCorrection");

    //! Virtual destructor
    virtual ~TwoBitCorrection ();

    //! Get the optimal value of the time series variance
    virtual double get_optimal_variance ();

    //! Set the sampling threshold as a fraction of the noise power
    void set_threshold (float threshold);

    //! Get the sampling threshold as a fraction of the noise power
    float get_threshold () const { return ja98.get_threshold(); }

    //! Set the digitization convention
    void set_table (TwoBitTable* table);

    //! Get the digitization convention
    const TwoBitTable* get_table () const;

    //! Return a pointer to a new instance of the appropriate sub-class
    static TwoBitCorrection* create (const BitSeries& input,
				     unsigned ndat_per_weight = 0,
				     float cutoff_rms = 3.0);

    //! Generate dynamic level setting and scattered power correction lookup
    void generate (std::vector<float>& dls, float* spc,
		   unsigned n_min, unsigned n_max, unsigned n_tot,
		   TwoBitTable* table, bool huge);

  protected:

    //! Build the two-bit correction look-up table and allocate histograms
    virtual void build ();

    //! Get the number of digitizer outputs in one byte
    virtual unsigned get_ndig_per_byte () const;

    //! Unpack a single polarization from raw into data
    virtual void dig_unpack (float* output_data,
			     const unsigned char* input_data, 
			     uint64 ndat,
			     unsigned digitizer,
			     unsigned* weights = 0,
			     unsigned nweights = 0);

    //! Two-bit conversion table generator
    Reference::To<TwoBitTable> table;

    //! Values used in Dynamic Level Setting
    std::vector< float > dls_lookup;

    //! Number of low-voltage states in a given byte
    std::vector< unsigned char > nlo_lookup;

    //! Build the number of low-voltage states lookup table
    virtual void nlo_build ();

  };
  
}

#endif
