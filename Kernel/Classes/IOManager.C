#include "dsp/IOManager.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"
#include "dsp/File.h"
#include "dsp/Unpacker.h"
#include "Error.h"

#include "genutil.h"

void dsp::IOManager::init()
{
  block_size = overlap = 0;
}

//! Constructor
dsp::IOManager::IOManager () : Input ("IOManager")
{
  init();
}

dsp::IOManager::~IOManager()
{
}

void dsp::IOManager::set_output (BitSeries* raw)
{
  if (verbose)
    cerr << "dsp::IOManager::set_output (BitSeries*) " << raw << endl;

  Input::set_output (raw);

  if (input)
    input -> set_output (raw);
  
  if (unpacker)
    unpacker -> set_input (raw);
}

void dsp::IOManager::set_output (TimeSeries* _data)
{
  if (verbose)
    cerr << "dsp::IOManager::set_output (TimeSeries*) " << _data << endl;

  data = _data;

  if (unpacker)
    unpacker -> set_output (_data);
}

//! Set the Input operator (should not normally need to be used)
void dsp::IOManager::set_input (Input* _input, bool set_params)
{
  input = _input;

  if (!input)
    return;

  if (set_params)  {
    input->set_block_size (block_size);
    input->set_overlap (overlap);
  }
  else {
    block_size = input->get_block_size ();
    overlap = input->get_overlap ();
  }

  if (output)
    input->set_output (output);

  name = "IOManager:" + input->get_name();

  set_unpacker ( Unpacker::create( input->get_info() ) );

  info = *( input->get_info() );

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


//! Set the number of time samples to load on each call to load_data
void dsp::IOManager::set_block_size (uint64 _block_size) 
{ 
  block_size = _block_size;
  if (input)
    input->set_block_size (block_size);
}
    
//! Set the number of time samples by which consecutive blocks overlap
void dsp::IOManager::set_overlap (uint64 _overlap) 
{ 
  overlap = _overlap;
  if (input)
    input->set_overlap (overlap);
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
void dsp::IOManager::open (const char* id, int bs_index) 
{
  try {

    set_input ( File::create(id,bs_index), true );

  } catch (Error& error) {
    throw error += "dsp::IOManager::open";
  }

}


//! The operation loads the next block of data and converts it to float_Stream
void dsp::IOManager::load (TimeSeries* _data)
{
  if (verbose)
    cerr << "dsp::IOManager::load (TimeSeries* = " << _data << ")" << endl;

  set_output (_data);

  operation ();
}


void dsp::IOManager::operation ()
{
  if (!output)
    set_output (new BitSeries);

  if (!input)
    throw Error (InvalidState, "dsp::IOManager::load", "no input");

  input->operate ();

  if (!data)
    return;

  if (!unpacker)
    throw Error (InvalidState, "dsp::IOManager::load", "no unpacker");

  unpacker->operate ();
}


void dsp::IOManager::load_data (BitSeries* data)
{
  if (!input)
    throw Error (InvalidState, "dsp::IOManager::load", "no input");

  input->set_output (data);
  input->operate ();
}


//! End of data
bool dsp::IOManager::eod()
{
  if (!input)
    return true;

  return input->eod();
}
    
//! Seek to the specified time sample
void dsp::IOManager::seek (int64 offset, int whence)
{
  if (!input)
    return;

  input->seek (offset, whence);
}

