#include "dsp/IOManager.h"
#include "dsp/Timeseries.h"
#include "dsp/File.h"
#include "dsp/Unpacker.h"
#include "dsp/TwoBitCorrection.h"
#include "Error.h"

#include "genutil.h"

void dsp::IOManager::init()
{
  block_size = overlap = 0;
  nsample = 512;
}

//! Constructor
dsp::IOManager::IOManager ()
{
  init();
}

dsp::IOManager::~IOManager()
{
}

//! Set the container from which input data will be read
void dsp::IOManager::set_raw (Timeseries* _raw)
{
  if (verbose)
    cerr << "IOManager::set_raw=" << _raw << endl;

  raw = _raw;

  if (unpacker)
    unpacker -> set_input (raw);
}

//! Set the Input operator (should not normally need to be used)
void dsp::IOManager::set_input (Input* _input)
{
  input = _input;

  if (input) {
    input->set_block_size (block_size);
    input->set_overlap (overlap);
  }
}

//! Set the Unpacker (should not normally need to be used)
void dsp::IOManager::set_unpacker (Unpacker* _unpacker)
{
  unpacker = _unpacker;

  TwoBitCorrection* tbc = dynamic_cast<TwoBitCorrection*> (_unpacker);

  if (tbc) {
    tbc -> set_cutoff_sigma (3.0);
    tbc -> set_nsample (nsample);
  }
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
    

//! Set the number of samples used to estimate undigitized power
void dsp::IOManager::set_nsample (int _nsample)
{ 
  nsample = _nsample;

  if (unpacker) {
    TwoBitCorrection* tbc = dynamic_cast<TwoBitCorrection*> (unpacker.get());
    if (tbc)
      tbc -> set_nsample (nsample);
  }
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
void dsp::IOManager::open (const char* id) 
{
  try {

    set_input ( File::create(id) );
    set_unpacker ( Unpacker::create (input->get_info()) );

  } catch (Error& error) {
    throw error += "IOManager::open";
  }

  info = *( input->get_info() );
}


//! The operation loads the next block of data and converts it to float_Stream
void dsp::IOManager::load (Timeseries* data)
{
  if (!input)
    throw string ("IOManager::load no input");

  if (!unpacker)
    throw string ("IOManager::load no unpacker");

  if (!raw)
    set_raw (new Timeseries);

  input->load (raw);

  if (verbose)
    cerr << "IOManager::load data=" << data << endl;

  unpacker->set_output (data);
  unpacker->operate();
}


void dsp::IOManager::load_data (Timeseries* data)
{
  throw string ("IOManager::load_data run-time error");
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
