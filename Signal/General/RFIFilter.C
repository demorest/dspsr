#include "dsp/RFIFilter.h"
#include "dsp/IOManager.h"
#include "dsp/TimeSeries.h"

dsp::RFIFilter::RFIFilter ()
{
  nchan_requested = 0;
  interval = 1.0;
  duty_cycle = 1.0;
}

dsp::RFIFilter::~RFIFilter ()
{
}

//! Create an RFI filter with nchan channels
void dsp::RFIFilter::match (const Observation* input, unsigned nchan)
{

}

//! Create an RFI filter with the same number of channels as Response
void dsp::RFIFilter::match (const Response* response)
{

}

//! Set the number of channels into which the band will be divided
void dsp::RFIFilter::set_nchan (unsigned nchan)
{

}

//! Set the interval over which the RFI mask will be calculated
void dsp::RFIFilter::set_update_interval (double seconds)
{

}

//! Set the fraction of the data used to calculate the RFI mask
void dsp::RFIFilter::set_duty_cycle (float cycle)
{

}

//! Set the source of the data
void dsp::RFIFilter::set_input (IOManager* _input)
{
  input = _input;
}

//! Set the buffer into which data will be read
void dsp::RFIFilter::set_buffer (TimeSeries* _buffer)
{
  buffer = _buffer;
}
