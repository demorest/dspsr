#include "IOManager.h"
#include "Timeseries.h"
#include "TwoBitCorrection.h"

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
  raw = _raw;

  if (converter)
    converter -> set_input (raw);
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

//! Set the conversion Operation (should not normally need to be used)
void dsp::IOManager::set_converter (Operation* _converter)
{
  converter = _converter;

  TwoBitCorrection* tbc = dynamic_cast<TwoBitCorrection*> (_converter);

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

  if (converter) {
    TwoBitCorrection* tbc = dynamic_cast<TwoBitCorrection*> (converter.get());
    if (tbc)
      tbc -> set_nsample (nsample);
  }
}

//! The operation loads the next block of data and converts it to float_Stream
void dsp::IOManager::load (Timeseries* data)
{
  if (!input || !converter)
    return;

  if (!raw)
    set_raw (new Timeseries);

  input->load (raw);

  converter->set_output (data);
  converter->operate();
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
void dsp::IOManager::seek (int64 offset, int whence = 0)
{
  if (!input)
    return;

  input->seek (offset, whence);
}
