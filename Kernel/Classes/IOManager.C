#include "dsp/IOManager.h"
#include "dsp/File.h"
#include "dsp/BitSeries.h"
#include "dsp/Unpacker.h"
#include "dsp/TimeSeries.h"

#include "Error.h"
//#include "genutil.h"

//! Constructor
dsp::IOManager::IOManager () : Operation ("IOManager")
{
}

dsp::IOManager::~IOManager()
{
}

void dsp::IOManager::set_output (BitSeries* raw)
{
  if (verbose)
    cerr << "dsp::IOManager::set_output (BitSeries*) " << raw << endl;

  output = raw;

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
void dsp::IOManager::set_input (Input* _input)
{
  input = _input;

  if (!input)
    return;

  if (output)
    input->set_output (output);

  name = "IOManager:" + input->get_name();

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
void dsp::IOManager::open (const string& id)
{
  try {

    set_input ( File::create(id) );

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

