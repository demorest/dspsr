#include "dsp/RFIFilter.h"

#include "dsp/WeightedTimeSeries.h"
#include "dsp/IOManager.h"
#include "dsp/Unpacker.h"
#include "dsp/Input.h"
#include "dsp/Bandpass.h"

#include "median_smooth.h"

dsp::RFIFilter::RFIFilter ()
{
  calculated = false;
  nchan_bandpass = 1024;
  interval = 1.0;
  duty_cycle = 1.0;

  maximum_block_size = 8*1024;
  median_window = 11;
}

dsp::RFIFilter::~RFIFilter ()
{
}

//! Create an RFI filter with nchan channels
void dsp::RFIFilter::match (const Observation* obs, unsigned nchan)
{
  MJD epoch = obs->get_start_time();
  MJD input_start = input->get_input()->get_info()->get_start_time();

  if (verbose)
    cerr << "dsp::RFIFilter::match obs epoch=" << epoch
	 << " current start=" << start_time << " end=" << end_time << endl;

  if (calculated && epoch > start_time && epoch < end_time) {
    if (verbose)
      cerr << "dsp::RFIFilter::match use current filter" << endl;
    return;
  }

  double offset = (epoch - input_start).in_seconds();
  cerr << "dsp::RFIFilter::match input offset=" << offset << "s" << endl;

  if (offset < 0)
    throw Error (InvalidState, "dsp::RFIFilter::match",
		 "requested epoch=" + epoch.printdays(6) +
		 " precedes input data start=" + input_start.printdays(6));

  uint64 ioffset = uint64 (offset / interval);
  offset = ioffset * interval;

  start_time = input_start + offset;
  end_time = start_time + interval;

  if (!bandpass)
    bandpass = new Bandpass;

  if (!buffer)
    buffer = new WeightedTimeSeries;

  if (!data)
    data = new Response;

  cerr << "making bandpass with nchan=" << nchan_bandpass << endl;

  bandpass->set_input (buffer);
  bandpass->set_output (data);
  bandpass->set_nchan (nchan_bandpass);
  bandpass->reset_output();

  uint64 position = input->get_input()->tell ();
  uint64 blocksz = input->get_input()->get_block_size ();
  uint64 overlap = input->get_input()->get_overlap ();
  TimeSeries* ptr = input->get_unpacker()->get_output();

  input->get_input()->seek (start_time);
  input->get_input()->set_block_size ( maximum_block_size );
  input->get_input()->set_overlap ( 0 );
  input->set_output (buffer);

  while (!input->get_input()->eod() &&
	 bandpass->get_integration_length() < interval) {
    input->operate ();
    bandpass->operate ();
  }

  input->get_input()->seek (position);
  input->get_input()->set_block_size (blocksz);
  input->get_input()->set_overlap (overlap);
  input->set_output (ptr);

  cerr << "computing mask" << endl;

  calculate (data);
}

void dsp::RFIFilter::calculate (Response* bp)
{
  unsigned nchan_bp = bp -> get_ndat();

  float* p0ptr = bp->get_datptr (0, 0);
  float* p1ptr = bp->get_datptr (0, 1);

  // form the total intensity bandpass
  vector<float> spectrum (nchan_bp);
  unsigned ichan=0;

  for (ichan=0; ichan < nchan_bp; ichan++)
    spectrum[ichan] = p0ptr[ichan]+p1ptr[ichan];

  fft::median_smooth (spectrum, median_window);

  double valsq = 0.0;
  for (ichan=0; ichan < nchan_bp; ichan++) {
    spectrum[ichan] -= (p0ptr[ichan]+p1ptr[ichan]);
    spectrum[ichan] *= spectrum[ichan];
    valsq += spectrum[ichan];
  }

  valsq /= nchan_bp;
  double rms = sqrt(valsq);

  cerr << "rms = " << rms << endl;

  resize (1, 1, nchan_bp, 2);

  float* ptr = get_datptr(0,0);

  for (ichan=0; ichan < nchan_bp; ichan++) {
    if (spectrum[ichan] > 16.0*valsq) {
      cerr << "ichan=" << ichan << endl;
      p0ptr[ichan] = p1ptr[ichan] = *ptr = 0.0;
    }
    else
      *ptr = 1;

    // imaginary
    ptr ++;
    *ptr = 0;
    ptr ++;
  }

    
  calculated = true;
}

//! Create an RFI filter with the same number of channels as Response
void dsp::RFIFilter::match (const Response* response)
{
  if (verbose)
    cerr << "dsp::RFIFilter::match Response nchan=" << response->get_nchan()
	 << " ndat=" << response->get_ndat() << endl;

  if ( get_nchan() == response->get_nchan() &&
       get_ndat() == response->get_ndat() ) {

    if (verbose)
      cerr << "dsp::RFIFilter::match Response already matched" << endl;
    return;

  }

  complex<float> one (1.0, 0.0);
  vector< complex<float> > phasors (response->get_nchan()*response->get_ndat(),
				    one);

  set (phasors);

  resize (1, response->get_nchan(),
	  response->get_ndat(), 2);


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

//! Set the buffer into which the spectra will be integrated [optional]
void dsp::RFIFilter::set_data (Response* _data)
{
  data = _data;
}

//! Set the tool used to compute the spectra [optional]
void dsp::RFIFilter::set_bandpass (Bandpass* _bandpass)
{
  bandpass = _bandpass;
}
