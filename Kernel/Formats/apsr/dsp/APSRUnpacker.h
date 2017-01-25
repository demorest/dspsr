//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/apsr/dsp/APSRUnpacker.h

#ifndef __APSRUnpacker_h
#define __APSRUnpacker_h

#include "dsp/UnpackerIterator.h"
#include "dsp/APSRIterator.h"
#include "dsp/excision_unpack.h"

#include "dsp/BitTable.h"
#include "dsp/Input.h"

namespace dsp
{

  // Implements ExcisionUnpacker interface for 2 4 and 8 bit unpackers
  template<class Parent, unsigned Nbit>
  class APSRUnpacker : public Parent
  {
  public:

    //! Constructor passes name to parent constructor
    APSRUnpacker (const char* name) : Parent (name) { }

    //! Template constructor passes name and argument to parent
    template<typename T>
    APSRUnpacker (const char* name, T arg) : Parent (name, arg) { }

    //! Match the iterator to the resolution
    void match_resolution (const Input* input)
    {
      Parent::match_resolution (input);
      iterator.init( input );
    }

    //! Decimated polyphase filterbank: one digitizer and complex
    unsigned get_ndim_per_digitizer () const { return 2; }

    bool matches (const Observation* observation)
    {
      return observation->get_machine() == "APSR"
        && observation->get_nbit() == Nbit
        && observation->get_state() == Signal::Analytic;
    }

    unsigned get_input_offset (unsigned idig) const
    {
      return idig * iterator.get_data_size();
    }

    //! When we try > 1 band, return input->get_nchan()
    unsigned get_input_incr () const { return 1; }

    class Implementation : public Unpacker::Iterator::Implementation
    {
    public:
      Implementation (const APSRIterator& copy,
		const unsigned char* base,
		unsigned idig)
	: iterator (copy)
      { iterator.set_base (base + idig * iterator.get_data_size()); }

      unsigned char get_value () const
      { return *iterator; }

      void increment ()
      { ++ iterator; }

      bool less_than (const unsigned char* ptr)
      { return iterator.ptr() < ptr; }

    protected:
      APSRIterator iterator;
    };

    //! Return the iterator for the specified frequency and polarization
    Unpacker::Iterator get_iterator (unsigned idig)
    {
      return Unpacker::Iterator
	( new Implementation (iterator, this->input->get_rawptr(), idig) );
    }

  protected:

    void dig_unpack (const unsigned char* input_data,
                     float* output_data,
                     uint64_t ndat,
                     unsigned long* hist,
                     unsigned* weights,
                     unsigned nweights)
    {
      iterator.set_base (input_data);
      Parent::excision_unpack (this->unpacker, iterator,
                               output_data, ndat, hist, weights, nweights);
    }

    // Iterator
    APSRIterator iterator;

  };

  // Manages an unpacker interface for 4 and 8 bit unpackers
  template<class U>
  class APSRExcision : public ExcisionUnpacker
  {
  public:

    //! Constructor
    APSRExcision (const char* name, BitTable* table) 
      : ExcisionUnpacker (name), unpacker (table)
    {
      this->ja98.set_threshold( table->get_nlow_threshold() );
    }

    //! Get the optimal value of the time series variance
    double get_optimal_variance ()
    {
      return unpacker.get_table()->get_optimal_variance();
    }

  protected:

    //! Unpacker used by APSRUnpacker
    U unpacker;

  };

}

#endif

