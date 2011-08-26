/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/UnpackerIterator.h"
#include "Error.h"

using namespace std;

//! Constructor
dsp::Unpacker::Unpacker (const char* name)
  : Transformation <BitSeries, TimeSeries> (name, outofplace) 
{
  output_order = TimeSeries::OrderFPT;
}

dsp::Unpacker * dsp::Unpacker::clone() const
{
  throw Error (InvalidState, "dsp::Unpacker::clone",
     "Not implemented in derived class");

  return 0;
}

void dsp::Unpacker::prepare ()
{
  if (verbose)
    cerr << "dsp::Unpacker::prepare" << endl;

  // set the Observation information
  output->Observation::operator=(*input);

  if (verbose)
    cerr << "dsp::Unpacker::prepare output start_time="
	 << output->get_start_time() << endl;
}

void dsp::Unpacker::reserve ()
{
  // resize the output
  output->set_order (output_order);

  if (verbose)
    cerr << "dsp::Unpacker::reserve input ndat=" << input->get_ndat() << endl;

  output->resize (input->get_ndat());
}

//! Return true if the unpacker support the specified output order
bool dsp::Unpacker::get_order_supported (TimeSeries::Order order) const
{
  // by default, only the current order is supported
  return order == output_order;
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::Unpacker::set_output_order (TimeSeries::Order order)
{
  if (order != output_order)
    throw Error (InvalidState, "dsp::Unpacker::set_output_order",
		 "unsupported output order");
}

//! Return true if the unpacker can operate on the specified device
bool dsp::Unpacker::get_device_supported (Memory* memory) const
{
  return memory == Memory::get_manager ();
}

//! Set the device on which the unpacker will operate
void dsp::Unpacker::set_device (Memory* memory)
{
  if (memory != Memory::get_manager ())
    throw Error (InvalidState, "dsp::Unpacker::set_device",
	               "unsupported device memory");
}

//! Initialize and resize the output before calling unpack
void dsp::Unpacker::transformation ()
{
  if (verbose)
    cerr << "dsp::Unpacker::transformation" << endl;;

  // set the Observation information
  prepare ();

  reserve ();

  // unpack the data
  unpack ();

  if (verbose)
    cerr << "dsp::Unpacker::tranformation TimeSeries book-keeping\n"
      "  input_sample=" << input->input_sample <<
      "  seek=" << input->get_request_offset() <<
      "  ndat=" << input->get_request_ndat() << endl;;

  // Set the input_sample attribute
  output->input_sample = input->input_sample;

  // The following lines deal with time sample resolution of the data source
  output->seek (input->get_request_offset());

  output->decrease_ndat (input->get_request_ndat());

  if (verbose)
    cerr << "dsp::Unpacker::transformation exit" << endl;;
}

//! Specialize the unpacker to the Observation
void dsp::Unpacker::match (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::Unpacker::match" << endl;
}

//! Return the iterator for the specified digitizer
dsp::Unpacker::Iterator dsp::Unpacker::get_iterator (unsigned idig)
{
  throw Error (InvalidState, "dsp::Unpacker::get_iterator",
	       "Iterator not implemented");
}
