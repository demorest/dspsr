#include <iostream>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "environ.h"
#include "OneBitCorrection.h"
#include "TimeSeries.h"
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
void dsp::OneBitCorrection::transformation ()
{
  if (input->get_nbit() != 1)
    throw_str ("OneBitCorrection::transformation input not 1-bit digitized");

  if (verbose)
    cerr << "Inside dsp::OneBitCorrection::transformation" << endl;;

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
    cerr << "dsp::OneBitCorrection::unpack" << endl;

  if (input->get_state() != Signal::Intensity)
    throw Error (InvalidState, "OneBitCorrection::unpack",
		 "input not total intensity");

  if (input->get_nbit() != 1)
    throw Error (InvalidState, "OneBitCorrection::unpack",
		 "input not 1-bit sampled");

  if( output->get_nbit() != 32 )
    throw Error(InvalidState,"dsp::OneBitCorrection::unpack()",
		"output doesn't have nbit=32");

  uint64 ndat = input->get_ndat();
  unsigned n_freq = input->get_nchan();

  if (n_freq % 32)
    throw Error (InvalidState, "OneBitCorrection::unpack",
		 "nchan=%d is not a multiple of 32", n_freq);

  if (verbose)
    cerr << "dsp::OneBitCorrection::unpack ndat="
	 << ndat << " n_freq=" << n_freq << endl;
  
  const uint32 * rawptr = (const uint32 *) input->get_rawptr();

#define MM 512
#define NN 16
   
  /*

    NN is the number of channels we do in one go- it should be a power of two between 1 and 32 as there are 32 channels in a uint32

    MM is the number of timesamples we do at any one time- when you have to write out to one of the output datptrs you may as well write several in one go as you have to copy a block of memory anyway.

    How the algorithm works is you want to output MM timesamples into each of NN output arrays.  For each timesample all NN channels are in one uint32, so you boolean 'and' ('&' operator) NN masks for that uint32.  Each time you do a boolean 'and' you'll either get zero (0.0) or non-zero (1.0).  You do one mask at a time for the MM values of one channel though.

   */
   
  unsigned masks[NN];
  
  for( unsigned i=0; i<NN; i++)
    masks[i] = unsigned(pow(unsigned(2),unsigned(i)));
  
  const register unsigned nskip = MM*n_freq/32;    
  
  for (unsigned ichan=0; ichan < n_freq; ichan+=NN) {
    register const unsigned shift = ichan % 32;
    
    const uint32* from = rawptr + ichan / 32;
    
      register float* intos[NN];
      for( unsigned i=0; i<NN; i++)
	intos[i] = output->get_datptr(ichan+i,0);
      
      register uint32 dat[MM];
      register const unsigned jump = n_freq/32;
      
      for (uint64 idat=0; idat < ndat; idat+=MM) {
	unsigned k = 0;
	for( unsigned j=0; j<MM; ++j, k+=jump)
	  dat[j] = from[k] >> shift;
	
	for( unsigned i=0; i<NN; ++i){
	  register float* fptr = intos[i]+idat;
	  
	  for( unsigned j=0; j<MM; ++j){
	    if( dat[j] & masks[i] )  fptr[j] = 1.0;
	    else	             fptr[j] = 0.0;
	  }
	}
	
	from += nskip;
      }
  }
  
  if( verbose )
    fprintf(stderr,"Bye from OneBitCorrection\n");
}

bool dsp::OneBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "PMDAQ"
    && observation->get_nbit() == 1;
}








