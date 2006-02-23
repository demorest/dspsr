//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/HistUnpacker.h,v $
   $Revision: 1.2 $
   $Date: 2006/02/23 22:22:08 $
   $Author: wvanstra $ */

#ifndef __HistUnpacker_h
#define __HistUnpacker_h

#include "dsp/Unpacker.h"
#include <vector>

namespace dsp {

  //! Base class of all unpackers that keep a histogram
  class HistUnpacker: public Unpacker {

  public:

    //! Maintain a diagnostic histogram of digitizer statistics
    static bool keep_histogram;

    //! Default constructor
    HistUnpacker (const char* name = "HistUnpacker");

    //! Virtual destructor
    virtual ~HistUnpacker ();

    //! Set the number of digitizers (histograms)
    virtual void set_ndig (unsigned ndig);
    //! Get the number of digitizers (histograms)
    unsigned get_ndig () const { return ndig; }

    //! Set the number of samples in the histogram
    virtual void set_nsample (unsigned nsample);
    //! Get the number of samples in the histogram
    unsigned get_nsample () const { return nsample; }

    //! Get the offset (number of floats) into output for the given digitizer
    virtual unsigned get_output_offset (unsigned idig) const;

    //! Get the output polarization for the given digitizer
    virtual unsigned get_output_ipol (unsigned idig) const;

    //! Get the output frequency channel for the given digitizer;
    virtual unsigned get_output_ichan (unsigned idig) const;

    //! Get the histogram for the given digitizer
    template <typename T>
    void get_histogram (vector<T>& data, unsigned idig) const;

    //! Get the centroid of the histogram for the given digitizer
    double get_histogram_mean (unsigned idig) const;

    //! Get the total number of samples in the histogram
    unsigned long get_histogram_total (unsigned idig) const;

    //! Reset histogram counts to zero
    void zero_histogram ();

  protected:

    //! Get the pointer to the histogram array
    unsigned long* get_histogram (unsigned idig)
	{ return &histograms[idig][0]; }

  private:

    //! Number of samples in the histogram
    unsigned nsample;

    //! Number of histograms
    unsigned ndig;

    //! Histograms of number of ones in nsample points
    vector< vector< unsigned long > > histograms;

    //! Resize the histograms vector to reflect attributes
    void resize ();

  };
  
}

template <typename T> 
void dsp::HistUnpacker::get_histogram (vector<T>& data, unsigned idig) const
{
  data.resize (nsample);

  for (unsigned i=0; i<nsample; i++)
    data[i] = T(histograms[idig][i]);
}

#endif
