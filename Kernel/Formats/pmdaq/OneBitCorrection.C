#include <iostream>
#include <assert.h>
#include <math.h>

#include "environ.h"
#include "OneBitCorrection.h"
#include "Timeseries.h"
#include "Error.h"

#include "genutil.h"

//! Null constructor
dsp::OneBitCorrection::OneBitCorrection (const char* _name)
  : Unpacker (_name)
{
}

dsp::OneBitCorrection::~OneBitCorrection ()
{
}

//! Initialize and resize the output before calling unpack
void dsp::OneBitCorrection::operation ()
{
  if (input->get_nbit() != 1)
    throw_str ("OneBitCorrection::operation input not 1-bit digitized");

  if (verbose)
    cerr << "Inside dsp::OneBitCorrection::operation" << endl;;

  // set the Observation information
  output->Observation::operator=(*input);

  // output will contain floating point values
  output->set_nbit (8 * sizeof(float));  // MXB K?

  // resize the output 
  output->resize (input->get_ndat());

  // unpack the data
  unpack ();
}

// MXB - this is where it all happens.
// Unpack the data into a 2D array with nfreq channels and ndat
// time samples.

void dsp::OneBitCorrection::unpack ()
{
  if (verbose)
    cerr << "dsp::OneBitCorrection::unpack input=" << input.get() << endl;;

  if (input->get_state() != Signal::Intensity)
    throw_str ("OneBitCorrection::unpack input not total intensity");

  if (verbose)
    cerr << "dsp::OneBitCorrection::unpack 0.1" << endl;

  if (input->get_nbit() != 1)
    throw_str ("OneBitCorrection::unpack input not 1-bit sampled");

  if (verbose)
    cerr << "dsp::OneBitCorrection::unpack 0.2" << endl;

  int64 ndat = input->get_ndat();
  int n_freq = input->get_nchan();

  if (verbose)
    cerr << "dsp::OneBitCorrection::unpack ndat="
	 << ndat << " n_freq=" << n_freq << endl;
  
  if (verbose)
    cerr << "Inside dsp::OneBitCorrection::unpack()2 " << endl;;
  const unsigned int * rawptr = (const unsigned int *)input->get_rawptr();

  //if (input->get_npol()!=1) 
  //  throw_str ("OneBitCorrection::unpack npol != 1, instead %d",
  //	       input_get_npol());

  if (verbose)
    cerr << "Inside dsp::OneBitCorrection::unpack()3" << endl;;

    //unsigned int * from = (unsigned int * ) rawptr;

    int n_samples = ndat; // * 8 / n_freq;

    if (verbose) cerr << " n_freq " <<n_freq<< " n_samples " <<
		   n_samples<< endl;

    const unsigned int mask = 0x01;
    // PMDAQ data comes in 8 1-bit sample chunks in reverse order??
    if (verbose) cerr << "unloading " << n_samples << " of data " << endl;
    float * base_address = output->get_datptr(0,0);
    int gap = output->get_datptr(1,0)-base_address;
    float * into;
    int pol_number;
    for (int i=0;i<n_samples;i++)
      {
      pol_number = 0;
      into = base_address + i;
      for (int j=0; j<n_freq/32;j++){
	for (int k=0;k<32;k++){
	  into += gap;
	  *into = (float) (((*rawptr)>>k) & mask);
	  pol_number++;
	}
	rawptr ++;
      }
      //      cout <<"."<< endl;
      }
}

bool dsp::OneBitCorrection::matches (const Observation* observation)
{
  return (observation->get_machine() == "PMDAQ" && observation->get_nbit()==1);
}
