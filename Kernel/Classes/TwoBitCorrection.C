/***************************************************************************
 *
 *   Copyright (C) 2002-2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

using namespace std;

#include "dsp/TwoBitCorrection.h"
#include "dsp/TwoBitTable.h"
#include "dsp/Input.h"
#include "dsp/WeightedTimeSeries.h"

#include "Error.h"
#include "ierf.h"

#include <assert.h>

// #define _DEBUG 1

bool dsp::TwoBitCorrection::change_levels = true;

//! Null constructor
dsp::TwoBitCorrection::TwoBitCorrection (const char* _name) 
  : ExcisionUnpacker (_name)
{
  // Sub-classes may re-define these
  set_ndat_per_weight (512);

  // Sub-classes must define this or set_table must be called
  table = NULL;

  // This is set in build()
  built = false;
}

dsp::TwoBitCorrection::~TwoBitCorrection ()
{
}

//! Get the optimal value of the time series variance
double dsp::TwoBitCorrection::get_optimal_variance ()
{
  if (change_levels)
    return ja98.A4( ja98.get_mean_Phi() );
  else
    return table->get_optimal_variance();
}


/*! By default, one digitizer is output in one byte */
unsigned dsp::TwoBitCorrection::get_ndig_per_byte () const
{ 
  return 1;
}

//! Set the cut off power for impulsive interference excision
void dsp::TwoBitCorrection::set_threshold (float threshold)
{
  if (threshold == ja98.get_threshold())
    return;

  if (verbose)
    cerr << "dsp::TwoBitCorrection::set_threshold = " << threshold << endl;

  ja98.set_threshold (threshold);
  built = false;
}


void dsp::TwoBitCorrection::set_table (TwoBitTable* _table)
{
  if (table.get() == _table)
    return;

  if (verbose)
    cerr << "dsp::TwoBitCorrection::set_table" << endl;

  table = _table;
  built = false;
}

//! Get the digitization convention
const dsp::TwoBitTable* dsp::TwoBitCorrection::get_table () const
{ 
  return table;
}



/* *************************************************************************
   TwoBitCorrection::build

   Generates a lookup table of output levels: y1 -> y4, for the range of 
   sample-statistics within the specified cutoff_sigma.

   This table may then be used to employ the dynamic level setting technique
   described by Jenet&Anderson in "Effects of Digitization on Nonstationary
   Stochastic Signals" for data recorded with CBR or CPSR.

   Where possible, references are made to the equations given in this paper,
   which are mostly found in Section 6.
   ********************************************************************** */

void dsp::TwoBitCorrection::build ()
{
  if (built)
    return;

  if (verbose) cerr << "dsp::TwoBitCorrection::build"
		    << " ppweight=" << get_ndat_per_weight()
		    << " cutoff=" << cutoff_sigma << "sigma\n";

  if (get_ndig()<1)
    throw Error (InvalidParam, "dsp::TwoBitCorrection::build",
		 "invalid number of digitizers=%d", get_ndig());

  if (!get_ndig_per_byte() ||
      table->get_values_per_byte() % get_ndig_per_byte())
    throw Error (InvalidParam, "dsp::TwoBitCorrection::build",
		 "invalid channels_per_byte=%d", get_ndig_per_byte());
 
  if (!table)
    throw Error (InvalidParam, "dsp::TwoBitCorrection::build",
		 "no TwoBitTable");

  set_limits ();

  unsigned n_range = nlow_max - nlow_min + 1;

  bool huge = (get_ndig_per_byte() == 1);

  unsigned size = table->get_unique_values();
  if (huge)
    size = BitTable::unique_bytes * table->get_values_per_byte();

  if (verbose) cerr << "dsp::TwoBitCorrection::build allocate buffers\n";
  dls_lookup.resize (n_range * size);

  generate (dls_lookup, 0, nlow_min, nlow_max, get_ndat_per_weight(), table, huge);

  zero_histogram ();

  nlo_build ();

  built = true;

  if (verbose) cerr << "dsp::TwoBitCorrection::build exits\n";

}

void dsp::TwoBitCorrection::nlo_build ()
{
  if (verbose)
    cerr << "dsp::TwoBitCorrection::nlo_build" << endl;

  nlo_lookup.resize (BitTable::unique_bytes);

  // flatten the table again (precision errors cause mismatch of lo_valsq)
  table->set_lo_val (1.0);
  table->rebuild();

  float lo_valsq = 1.0;

  for (unsigned byte = 0; byte < BitTable::unique_bytes; byte++)
  {
    nlo_lookup[byte] = 0;
    const float* fourvals = table->get_values (byte);

    for (unsigned ifv=0; ifv<table->get_values_per_byte(); ifv++)
      if (fourvals[ifv]*fourvals[ifv] == lo_valsq)
	nlo_lookup[byte] ++;
  }
}

void dsp::TwoBitCorrection::generate (vector<float>& dls, float* spc, 
				      unsigned nlow_min,
				      unsigned nlow_max,
				      unsigned n_tot,
				      TwoBitTable* table, bool huge)
{
  unsigned offset = 0;

  for (unsigned nlo=nlow_min; nlo <= nlow_max; nlo++)
  {
    /* Refering to JA98, nlo is the number of samples between x2 and x4, 
       and p_in is the left-hand side of Eq.44 */
    float p_in = (float) nlo / (float) n_tot;

    if (change_levels)
    {
      ja98.set_Phi (p_in);
      
      table->set_lo_val ( ja98.get_lo() );
      table->set_hi_val ( ja98.get_hi() );
      table->rebuild();
    }
    
    if (huge)
    {
      /* Generate the 256 sets of four output floating point values
	 corresponding to each byte */
      table->generate ( &(dls[offset]) );
      offset += BitTable::unique_bytes * table->get_values_per_byte();
    }
    else
    {
      // Generate the four output levels corresponding to each 2-bit number
      table->generate_unique_values ( &(dls[offset]) );
      offset += table->get_unique_values();
    }

    if (change_levels && spc)
    {
      *spc = ja98.get_A();
      spc ++;
    }

  }

  assert (offset == dls.size());
}

void dsp::TwoBitCorrection::dig_unpack (float* output_data,
					const unsigned char* input_data, 
					uint64 nfloat,
					unsigned digitizer,
					unsigned* weights,
					unsigned nweights)
{
  unsigned ndig = get_ndig_per_byte();

  if (ndig != 1)
    throw Error (InvalidState, "dsp::TwoBitCorrection::dig_unpack",
		 "number of digitizers per byte = %d must be == 1", ndig);

#ifndef _DEBUG
  if (verbose)
#endif
    cerr << "dsp::TwoBitCorrection::dig_unpack in=" << (void*) input_data
	 << " out=" << output_data << " nfloat=" << nfloat << "\n\t"
	 << " digitizer=" << digitizer << " weights=" << weights 
	 << " nweights=" << nweights << endl;

  // 4 floating-point samples per byte
  const unsigned samples_per_byte = table->get_values_per_byte();

  // 4*256 floating-point samples for all unique bytes
  const unsigned lookup_block_size =
    samples_per_byte * BitTable::unique_bytes;

  unsigned long* hist = 0;
  if (keep_histogram)
    hist = get_histogram (digitizer);

  bool data_are_complex = get_ndim_per_digitizer() == 2;

  unsigned nfloat_per_weight = get_ndat_per_weight()*get_ndim_per_digitizer();

  double f_weights = double(nfloat) / double(nfloat_per_weight);
  unsigned long n_weights = (unsigned long) ceil (f_weights);

  assert (n_weights*nfloat_per_weight >= nfloat);

  if (weights && n_weights > nweights)
    throw Error (InvalidParam, "dsp::TwoBitCorrection::dig_unpack",
		 "weights array size=%d < nweights=%d", nweights, n_weights);

  uint64 bytes_left = nfloat / samples_per_byte;
  uint64 bytes_per_weight = nfloat_per_weight / samples_per_byte;
  uint64 bytes_to_unpack = bytes_per_weight;
  unsigned bt;
  unsigned pt;

  unsigned input_incr = get_input_incr ();
  unsigned output_incr = get_output_incr ();

#ifdef _DEBUG
  cerr << "dsp::TwoBitCorrection::dig_unpack output_incr=" << output_incr 
       << " input_incr=" << input_incr << endl;
#endif

  float* section = 0;
  float* fourval = 0;

// #define APSR_DEV 1

#ifdef APSR_DEV

  unsigned sample_resolution = get_input()->get_loader()->get_resolution();

  unsigned byte_resolution = get_input()->get_nbytes(sample_resolution);

  unsigned dig_bytes = byte_resolution / get_ndig();

#ifdef _DEBUG
  cerr << "dsp::TwoBitCorrection::dig_unpack nweight=" << n_weights
       << " ndig=" << get_ndig() << " dig_bytes=" << dig_bytes << endl;
#endif

  const unsigned char* end_of_packet = 0;
  if (dig_bytes > 1)
    end_of_packet = input_data + dig_bytes;

  const unsigned char* packet_ptr = end_of_packet;

#endif

  const unsigned char* input_data_ptr = input_data;

  const unsigned n_lo_max = get_ndat_per_weight();

  for (unsigned long wt=0; wt<n_weights; wt++)
  {
#ifdef _DEBUG
    cerr << wt << " ";
#endif

    if (bytes_to_unpack > bytes_left)
    {
      input_data_ptr -= (bytes_per_weight - bytes_left) * input_incr;
      // cerr << "off " << (bytes_per_weight - bytes_left) * input_incr << endl;
      bytes_to_unpack = bytes_left;
    }

    // calculate the weight based on the last ndat_per_weight pts
    unsigned n_lo = 0;
    for (bt=0; bt<bytes_per_weight; bt++)
    {
#if 0
      if (input_data_ptr >= end_of_buffer)
      {
        cerr << "past end of buffer on wt=" << wt << " nwt=" << n_weights 
             << " bt=" << bt << " bytes_left=" << bytes_left << " rem=" << bytes_left - bt << endl;
        exit(-1);
      }
#endif

      n_lo += nlo_lookup [*input_data_ptr];
      input_data_ptr += input_incr;

#ifdef APSR_DEV

#ifdef _DEBUG2
cerr << "data_ptr=" << (void*)input_data_ptr << " pack_ptr=" << (void*)packet_ptr << " incr=" << input_incr << endl;
#endif

      if (input_data_ptr == packet_ptr)
      {
        input_data_ptr += dig_bytes;
        packet_ptr += byte_resolution;
      }
#endif

    }

    // if data are complex, quickly divide n_lo by two
    if (data_are_complex)
      n_lo >>= 1;

    if (hist && n_lo < n_lo_max)
      hist [n_lo] ++;

    input_data_ptr = input_data;

#ifdef APSR_DEV
    packet_ptr = end_of_packet;
#endif

    // test if the number of low voltage states is outside the
    // acceptable limit or if this section of data has been previously
    // flagged bad (for example, due to bad data in the other polarization)
    if ( n_lo<nlow_min || n_lo>nlow_max || (weights && weights[wt] == 0) )
    {
#ifdef _DEBUG2
       cerr << "w[" << wt << "]=0 ";
#endif
      
      if (weights)
        weights[wt] = 0;
      
      // reduce the risk of other functions accessing un-initialized 
      // segments of the array
      for (bt=0; bt<bytes_to_unpack; bt++)
      {
        for (pt=0; pt<samples_per_byte; pt++)
        {
	  *output_data = 0.0;
	  output_data += output_incr;
        }
        input_data_ptr += input_incr;

#ifdef APSR_DEV
        if (input_data_ptr == packet_ptr)
        {
          input_data_ptr += dig_bytes;
          packet_ptr += byte_resolution;
        }
#endif
      }
    }
    else
    {
      section = &(dls_lookup[0]) + (n_lo-nlow_min) * lookup_block_size;
      
      for (bt=0; bt<bytes_to_unpack; bt++)
      {
	fourval = section + unsigned(*input_data_ptr) * samples_per_byte;
	for (pt=0; pt<samples_per_byte; pt++)
        {
	  *output_data = fourval[pt];
#ifdef _DEBUG2
          cerr << "b: " << *output_data << endl;
#endif
	  output_data += output_incr;
	}

	input_data_ptr += input_incr;

#ifdef APSR_DEV
        if (input_data_ptr == packet_ptr)
        {
          input_data_ptr += dig_bytes;
          packet_ptr += byte_resolution;
        }
#endif

      }

      if (weights)
	weights[wt] = n_lo;
      
    }

    bytes_left -= bytes_to_unpack;
    input_data = input_data_ptr;

#ifdef APSR_DEV
    end_of_packet = packet_ptr;
#endif

  }

#ifdef _DEBUG
  cerr << "DONE!" << endl;
#endif
  
}

