/***************************************************************************
 *
 *   Copyright (C) 2002-2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "environ.h"

#include "dsp/IOManager.h"
#include "dsp/File.h"
#include "dsp/BitSeries.h"
#include "dsp/Unpacker.h"
#include "dsp/TimeSeries.h"

#include "templates.h"
#include "Error.h"

using namespace std;

//! Constructor
dsp::IOManager::IOManager () : Operation ("IOManager")
{
  maximum_RAM = 0;
  minimum_RAM = 0;
  copies = 1;
}

dsp::IOManager::~IOManager()
{
}

void dsp::IOManager::set_maximum_RAM (uint64_t max)
{
  maximum_RAM = max;
}

//! Set the minimum RAM usage constraint in set_block_size
void dsp::IOManager::set_minimum_RAM (uint64_t min)
{
  minimum_RAM = min;
}

//! Set the number of copies of data constraint in set_block_size
void dsp::IOManager::set_copies (unsigned nbuf)
{
  copies = nbuf;
}

//! Set the scratch space
void dsp::IOManager::set_scratch (Scratch* s)
{
  if (verbose)
    cerr << "dsp::IOManager::set_scratch" << endl;

  Operation::set_scratch( s );

  if (input && !input->context)
    input->set_scratch( s );

  if (unpacker)
    unpacker->set_scratch( s );
}

//! Set verbosity ostream
void dsp::IOManager::set_cerr (std::ostream& os) const
{
  Operation::set_cerr( os );

  if (verbose)
    cerr << "dsp::IOManager::set_cerr" << endl;

  if (input && !input->context)
    input->set_cerr( os );

  if (output)
    output->set_cerr( os );

  if (unpacker)
    unpacker->set_cerr( os );
}

void dsp::IOManager::set_output (BitSeries* raw)
{
  if (verbose)
    cerr << "dsp::IOManager::set_output (BitSeries*) " << raw << endl;

  output = raw;

  if (unpacker)
  {
    if (verbose)
      cerr << "dsp::IOManager::set_output call Unpacker::set_input" << endl;
    unpacker -> set_input (raw);
  }
}

void dsp::IOManager::set_output (TimeSeries* _data)
{
  if (verbose)
    cerr << "dsp::IOManager::set_output (TimeSeries*) " << _data << endl;

  data = _data;

  if (unpacker)
  {
    if (verbose)
      cerr << "dsp::IOManager::set_output call Unpacker::set_output" << endl;
    unpacker -> set_output (_data);
  }
}

//! Set the Input operator (should not normally need to be used)
void dsp::IOManager::set_input (Input* _input)
{
  input = _input;

  if (!input)
    return;

  name = "IOManager:" + input->get_name();

  if (!unpacker || !unpacker->matches (input->get_info()))
    set_unpacker ( Unpacker::create( input->get_info() ) );
}

const dsp::Observation* dsp::IOManager::get_info () const
{
  return input->get_info();
}

dsp::Observation* dsp::IOManager::get_info ()
{
  return input->get_info();
}

//! Return pointer to the appropriate Input
const dsp::Input* dsp::IOManager::get_input () const 
{
  return input;
}

dsp::Input* dsp::IOManager::get_input ()
{
  return input;
}


//! Set the Unpacker (should not normally need to be used)
void dsp::IOManager::set_unpacker (Unpacker* _unpacker)
{
  unpacker = _unpacker;

  if (unpacker)  {
    if (output)
      unpacker -> set_input (output);
    if (data)
      unpacker -> set_output (data);
  }
}

const dsp::Unpacker* dsp::IOManager::get_unpacker () const 
{ 
  return unpacker;
}

dsp::Unpacker* dsp::IOManager::get_unpacker ()
{
  return unpacker;
}



 
//! Prepare the appropriate Input and Unpacker
/*!

  \param id string containing the id of the data source.  The source
  id may be a:
  <UL>
  <LI> filename
  <LI> a comma separated list of filenames to be treated as one observation
  <LI> a string of the form "IPC:xx", where "xx" is a shared memory key
  </UL>

  \pre This function is not fully implemented.
*/
void dsp::IOManager::open (const string& id) try
{
  set_input ( File::create(id) );
}
catch (Error& error)
{
  throw error += "dsp::IOManager::open";
}

//! The operation loads the next block of data and converts it to float_Stream
void dsp::IOManager::load (TimeSeries* _data)
{
  if (verbose)
    cerr << "dsp::IOManager::load (TimeSeries* = " << _data << ")" << endl;

  set_output (_data);

  operation ();
}

void dsp::IOManager::prepare ()
{
  if (verbose)
    cerr << "dsp::IOManager::prepare" << endl;

  if (!output)
    set_output (new BitSeries);

  input->set_output( output );

  input->prepare();
  unpacker->prepare();
  if (post_load_operation)
    post_load_operation->prepare ();


  prepared = true;
}

void dsp::IOManager::reserve ()
{
  if (verbose)
    cerr << "dsp::IOManager::reserve" << endl;

  input->reserve( output );
  unpacker->reserve();
  if (post_load_operation)
    post_load_operation->reserve ();

}

void dsp::IOManager::add_extensions (Extensions* ext)
{
  if (input)
    input->add_extensions (ext);
  if (unpacker)
    unpacker->add_extensions (ext);
  if (post_load_operation)
    post_load_operation->add_extensions (ext);

}

void dsp::IOManager::combine (const Operation* other)
{
  Operation::combine (other);

  const IOManager* like = dynamic_cast<const IOManager*>( other );
  if (!like)
    return;

  input->combine (like->input);
  unpacker->combine (like->unpacker);
  if (post_load_operation)
    post_load_operation->combine (like->post_load_operation);

}

void dsp::IOManager::reset ()
{
  Operation::reset ();

  input->reset ();
  unpacker->reset ();
  if (post_load_operation)
    post_load_operation->reset ();
}

void dsp::IOManager::report () const
{
  if (input)
    input->report ();
  if (unpacker)
    unpacker->report ();
  if (post_load_operation)
    post_load_operation->report ();

}

void dsp::IOManager::operation ()
{
  if (!output)
    set_output (new BitSeries);

  input->load (output);

  if (post_load_operation)
  {
    if (verbose)
      cerr << "dsp::IOManager::operation post_load_operation->operate()" << endl;
    post_load_operation->operate ();
  }

  if (!data)
    return;

  unpacker->operate ();
}

void dsp::IOManager::set_post_load_operation (Operation * op)
{
  if (verbose)
    cerr << "dsp::IOManager::set_post_load_operation(" << op << ")" << endl;
  post_load_operation = op;
}

uint64_t dsp::IOManager::set_block_size (uint64_t minimum_samples)
{
  if (verbose)
    cerr << "dsp::IOManager::set_block_size minimum_samples=" 
         << minimum_samples << endl;

  /*
    This simple calculation of the maximum block size does not
    consider the RAM required for FFT plans, etc.
  */

  unsigned resolution = input->get_resolution();

  if (verbose)
    cerr << "dsp::IOManager::set_block_size input resolution="
         << resolution << endl;

  unpacker->match_resolution (input);
  if (unpacker->get_resolution())
  {
    resolution = unpacker->get_resolution();
    if (verbose)
      cerr << "dsp::IOManager::set_block_size unpacker resolution="
           << resolution << endl;
  }

  // ensure that the block size is a multiple of four
  if (resolution % 4)
  {
    if (resolution % 2 == 0)
      resolution *= 2;
    else
      resolution *= 4;
  }

  Observation* info = get_info();

  unsigned nbit  = info->get_nbit();
  unsigned ndim  = info->get_ndim();
  unsigned npol  = info->get_npol();
  unsigned nchan = info->get_nchan();

  // each nbit number will be unpacked into a float
  double nbyte = double(nbit)/8 + copies * sizeof(float);

  if (verbose)
    cerr << "dsp::IOManager::set_block_size copies=" << copies
         << " nbit=" << nbit << " nbyte=" << nbyte << endl;

  double nbyte_dat = nbyte * ndim * npol * nchan;

  uint64_t block_size = multiple_greater (minimum_samples, resolution);

  if (verbose)
    cerr << "dsp::IOManager::set_block_size required block_size="
         << block_size << endl;

  if (minimum_RAM)
  {
    uint64_t size = (uint64_t(minimum_RAM/nbyte_dat)/resolution) * resolution;
    if (verbose)
      cerr << "dsp::IOManager::set_block_size"
	" minimum block_size=" << size << endl;

    block_size = std::max (block_size, size);
  }

  if (maximum_RAM)
  {
    block_size = (uint64_t(maximum_RAM / nbyte_dat) / resolution) * resolution;
    if (verbose)
      cerr << "dsp::IOManager::set_block_size maximum block_size="
           << block_size << endl;
  }
 
  float megabyte = 1024 * 1024;

  if (block_size < minimum_samples)
  {
    float blocks = float(minimum_samples)/float(resolution);
    float min_ram = ceil (blocks) * resolution * nbyte_dat;

    if (verbose)
      cerr << "dsp::IOManager::set_block_size insufficient RAM" << endl;

    throw Error (InvalidState, "dsp::IOManager::set_block_size",
		 "insufficient RAM: limit=%g MB -> block="UI64" samples\n\t"
		 "require="UI64" samples -> \"-U %g\" on command line",
		 float(maximum_RAM)/megabyte, block_size,
		 minimum_samples, min_ram/megabyte);
  }

  if (input->get_overlap())
  {
    unsigned overlap = input->get_overlap();

    double parts = (block_size - overlap) / (minimum_samples - overlap);

    if (verbose)
      cerr << "dsp::IOManager::set_block_size input"
              " overlap=" << overlap << " parts=" << parts << endl;

    uint64_t block_resize = unsigned(parts)*(minimum_samples-overlap) + overlap;

    if (verbose)
      cerr << "dsp::IOManager::set_block_size old=" << block_size
           << " new=" << block_resize << endl;

    block_size = block_resize;
  }

  input->set_block_size ( block_size );

  return uint64_t( block_size * nbyte_dat );
}

void dsp::IOManager::set_overlap (uint64_t overlap)
{
  if (verbose)
    cerr << "dsp::IOManager::set_overlap request overlap=" << overlap << endl;

  unsigned resolution = input->get_resolution();

  if (verbose)
    cerr << "dsp::IOManager::set_overlap input resolution="
         << resolution << endl;

  if (unpacker->get_resolution())
  {
    resolution = unpacker->get_resolution();
    if (verbose)
      cerr << "dsp::IOManager::set_overlap unpacker resolution="
           << resolution << endl;
  }

  // ensure that the block size is a multiple of four
  if (resolution % 4)
  {
    if (resolution % 2 == 0)
      resolution *= 2;
    else
      resolution *= 4;
  }

  overlap = multiple_greater (overlap, resolution);

  if (verbose)
    cerr << "dsp::IOManager::set_overlap require overlap=" << overlap << endl;
    
  input->set_overlap( overlap );
}

