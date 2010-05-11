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
#include "dsp/StepIterator.h"
#include "dsp/excision_unpack.h"

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
  not_built ();
}

void dsp::TwoBitCorrection::set_table (TwoBitTable* _table)
{
  if (table.get() == _table)
    return;

  if (verbose)
    cerr << "dsp::TwoBitCorrection::set_table" << endl;

  table = _table;
  not_built ();
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

  ExcisionUnpacker::build ();

  TwoBitLookup* lookup = get_unpacker ();

  lookup->set_nlow_min (nlow_min);
  lookup->set_nlow_max (nlow_max);
  lookup->set_ndat (get_ndat_per_weight());
  lookup->set_ndim (get_ndim_per_digitizer());

  lookup->lookup_build (table, &ja98);

  if (verbose) cerr << "dsp::TwoBitCorrection::build exits\n";
}

void dsp::TwoBitCorrection::dig_unpack (const unsigned char* input_data, 
					float* output_data,
					uint64_t nfloat,
					unsigned long* hist,
					unsigned* weights,
					unsigned nweights)
{
  StepIterator<const unsigned char> iterator (input_data);
  iterator.set_increment ( get_input_incr() );

  ExcisionUnpacker::excision_unpack (unpacker, iterator,
				     output_data, nfloat,
				     hist, weights, nweights);

}

