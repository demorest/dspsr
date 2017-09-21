//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/HistUnpacker.h

#ifndef __HistUnpacker_h
#define __HistUnpacker_h

#include "dsp/Unpacker.h"
#include <vector>

namespace dsp {

  //! Bit unpacker that keeps a histogram and optimal statistics
  class HistUnpacker: public Unpacker
  {

  public:

    //! Maintain a diagnostic histogram of digitizer statistics
    static bool keep_histogram;

    //! Default constructor
    HistUnpacker (const char* name = "HistUnpacker");

    //! Virtual destructor
    virtual ~HistUnpacker ();

    //! If Operation is a HistUnpacker, integrate its histograms
    void combine (const Operation*);

    //! Reset the histograms
    void reset ();

    //! Get the optimal value of the time series variance
    virtual double get_optimal_variance ();

    //! Set the number of digitizers (histograms)
    virtual void set_ndig (unsigned ndig);
    //! Get the number of digitizers (histograms)
    virtual unsigned get_ndig () const;

    //! Get the dimension of the digitizer outputs (real or complex)
    virtual unsigned get_ndim_per_digitizer () const;

    //! Set the number of states in the histogram
    virtual void set_nstate (unsigned nstate);
    //! Get the number of states in the histogram
    unsigned get_nstate () const { return nstate; }

    //! Get the offset (number of floats) into output for the given digitizer
    virtual unsigned get_output_offset (unsigned idig) const;

    //! Get the output polarization for the given digitizer
    virtual unsigned get_output_ipol (unsigned idig) const;

    //! Get the output frequency channel for the given digitizer;
    virtual unsigned get_output_ichan (unsigned idig) const;

    //! Get the histogram for the specified digitizer
    virtual void get_histogram (std::vector<unsigned long>&, unsigned) const;

    //! Get the histogram for the given digitizer
    template <typename T>
    void get_histogram (std::vector<T>& data, unsigned idig) const;

    //! Get the pointer to the histogram array
    unsigned long* get_histogram (unsigned idig, unsigned expect = 0);
    const unsigned long* get_histogram (unsigned idig) const;

    //! Get the centroid of the histogram for the given digitizer
    double get_histogram_mean (unsigned idig) const;

    //! Get the total number of samples in the histogram
    unsigned long get_histogram_total (unsigned idig) const;

    //! Reset histogram counts to zero
    void zero_histogram ();

  protected:

    //! Compute the default number of digitizers
    virtual void set_default_ndig ();

    void set_nstate_internal (unsigned _nstate);
    unsigned get_nstate_internal () const;

  private:

    //! Number of states in the histogram
    unsigned nstate;

    //! Number of states in the internal representation of the histogram
    unsigned nstate_internal;

    //! Number of histograms
    unsigned ndig;

    //! Histograms of nstate (or nstate_internal, if set) states
    std::vector< std::vector< unsigned long > > histograms;

    //! Resize the histograms vector to reflect attributes
    void resize ();

    bool resize_needed;
  };
  
}

template <typename T> void
dsp::HistUnpacker::get_histogram (std::vector<T>& data, unsigned idig) const
{
  std::vector<unsigned long> hist;
  get_histogram (hist, idig);

  data.resize( hist.size() );

  for (unsigned i=0; i<hist.size(); i++)
    data[i] = T(hist[i]);
}

#endif
