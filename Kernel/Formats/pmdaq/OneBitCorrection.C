/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/OneBitCorrection.h"
#include "dsp/PMDAQ_Extension.h"

using namespace std;

//! Null constructor
dsp::OneBitCorrection::OneBitCorrection (const char* _name)
  : Unpacker (_name)
{
  first_chan = 0;
  end_chan = 99999;

  generate_lookup();
}

void dsp::OneBitCorrection::generate_lookup(){
  unsigned char i = 0;

  for( i=0; i<255; ++i){
    for( unsigned char j=0; j<8; ++j){
      if( i & (1 << j) )  lookup[i*8+j] = 1.0;
      else                lookup[i*8+j] = 0.0;
    }
  }

  i = 255;
  for( unsigned char j=0; j<8; ++j){
    if( i & (1 << j) )  lookup[i*8+j] = 1.0;
    else                lookup[i*8+j] = 0.0;
  }


}

dsp::OneBitCorrection::~OneBitCorrection (){ }

unsigned dsp::OneBitCorrection::get_ndig () const
{
  return input->get_nchan();
}

//! Initialize and resize the output before calling unpack
void dsp::OneBitCorrection::transformation ()
{
  if (input->get_nbit() != 1)
    throw Error(InvalidState,"dsp::OneBitCorrection::transformation()",
		"input not 1-bit digitized");

  if (verbose)
    cerr << "Inside dsp::OneBitCorrection::transformation" << endl;

#if FIX_THIS
  if( input->has<PMDAQ_Extension>() ){
    set_first_chan( input->get<PMDAQ_Extension>()->get_chan_begin() );
    set_end_chan( input->get<PMDAQ_Extension>()->get_chan_end() );
  }
#endif

  unsigned end_channel = min(input->get_nchan(),end_chan);
  unsigned required_nchan = end_channel - first_chan;

  // Make sure output nchan doesn't change if it's already been set:
  {
    Reference::To<Observation> dummy = new Observation(*input);
    dummy->set_nchan( required_nchan );
    output->Observation::operator=(*dummy);
  }

  // output will contain floating point values
  output->set_nbit (8 * sizeof(float));
  
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

  unsigned ndat = unsigned(input->get_ndat());
  unsigned n_freq = output->get_nchan();

  if (n_freq % 32)
    throw Error (InvalidState, "OneBitCorrection::unpack",
		 "n_freq=%d is not a multiple of 32", n_freq);
  if( first_chan % 32)
    throw Error (InvalidState, "OneBitCorrection::unpack",
		 "first_chan=%d is not a multiple of 32", first_chan);

  if (verbose)
    cerr << "dsp::OneBitCorrection::unpack ndat="
	 << ndat << " n_freq=" << n_freq << endl;

  const uint32_t * rawptr = (const uint32_t *) input->get_rawptr();

  // Note that algorithm 1 loses ndat%MM samples
  //  unsigned algorithm = 1;

  RealTimer rt0;

  //  if( algorithm==1 ){
    rt0.start();

#define MM 512
#define NN 16

  /*

    NN is the number of channels we do in one go- it should be a power of two between 1 and 32 as there are 32 channels in a uint32_t

    MM is the number of timesamples we do at any one time- when you have to write out to one of the output datptrs you may as well write several in one go as you have to copy a block of memory anyway.

    How the algorithm works is you want to output MM timesamples into each of NN output arrays.  For each timesample all NN channels are in one uint32_t, so you bitwise 'and' ('&' operator) NN masks for that uint32_t.  Each time you do a bitwise 'and' you'll either get zero (0.0) or non-zero (1.0).  You do one mask at a time for the MM values of one channel though.



   */
 
  uint32_t masks[NN];
  
  for( uint32_t i=0; i<NN; i++)
    masks[i] = uint32_t(1) << i;
  
  // 'nskip' is the number of uint32_t's you want to skip over in the input stream every time you do a new set of MM timesamples 
  register const unsigned nskip = MM*input->get_nchan()/32;    
  // 'jump' is the number of uint32_t's between timesamples of the same channel
  register const unsigned jump = input->get_nchan()/32;

  for (unsigned ichan=first_chan; ichan < n_freq+first_chan; ichan+=NN) {
    // e.g. if NN is 16 then 'shift' is whether you want the first 16 or the second 16 bits in each uint32_t
    register const uint32_t shift = ichan % 32;
    
    const uint32_t* from = rawptr + ichan / 32;
    
    register float* intos[NN];
    for( unsigned i=0; i<NN; i++)
      intos[i] = output->get_datptr(i+ichan-first_chan,0);
    
    register uint32_t dat[MM];
    
    // NOTE: because idat increments in steps of MM, ndat%MM samples are lost!
    for (unsigned idat=0; idat < ndat; idat+=MM) {
      // Grab MM uint32_t's, shift out the bits we're not interested in, and store those MM uint32_t's in a contiguous array
      unsigned k = 0;
      for( unsigned j=0; j<MM; ++j, k+=jump)
	dat[j] = from[k] >> shift;
      
      // Do the work for the NN channels in the MM-sized array, 'dat'.
      for( unsigned i=0; i<NN; ++i){
	register float* fptr = intos[i] + idat;
	for( unsigned j=0; j<MM; ++j){
	  if( dat[j] & masks[i] )  fptr[j] = 1.0;
	  else	                   fptr[j] = 0.0;
	}
      }
      from += nskip;
    }

  }
  rt0.stop();
  if( verbose )
    fprintf(stderr,"alg 1: %f secs\n",rt0.get_total());

  /*
  else if( algorithm == 2 ){
    fprintf(stderr,"hi2\n");

    //
    // Reorder as TimeSeries
    //

    RealTimer rt1;
    rt1.start();

    register const unsigned jump = input->get_nchan()/32;
    
    uint32_t* onebit_ts = new uint32_t[ndat*jump];
    for( unsigned i=0; i<ndat*jump; i++)
      onebit_ts[i] = 0;
    
    uint32_t masks[32];
    for( unsigned i=0; i<32; i++)
      masks[i] = 1 << i;

    uint32_t ndat_on_32 = ndat/32;

    //for( unsigned idat=0; idat<ndat; idat++){
    //unsigned dat_offset = idat*jump;
    //
    //for( unsigned ichan=0; ichan<input->get_nchan(); ichan++)
    //onebit_ts[ichan*ndat_on_32 + idat/32] |= rawptr[ichan/32 + dat_offset] & masks[ichan%32];
    //}

    for (unsigned ichan=0; ichan < n_freq; ichan+=NN) {
      register const uint32_t shift = ichan % 32;
      
      const uint32_t* from = rawptr + ichan / 32;
      
      register uint32_t* intos[NN/32];
      for( unsigned i=0; i<NN/32; i++)
	intos[i] = onebit_ts[(ichan+i)*ndat/32];
      
      register uint32_t dat[MM];
      
      for (unsigned idat=0; idat < ndat; idat+=MM) {
	unsigned k = 0;
	for( unsigned j=0; j<MM; ++j, k+=jump)
	  dat[j] = from[k] >> shift;
	
	for( unsigned i=0; i<NN; ++i){
	  register float* fptr = intos[i] + idat;
	  for( unsigned j=0; j<MM; ++j){
	    if( dat[j] & masks[i] )  fptr[j] = 1.0;
	    else	             fptr[j] = 0.0;
	  }
	}
	from += nskip;
      }
      
    }

    rt1.stop();

    RealTimer rt2;
    rt2.start();
      
    //
    // Expand to a 32-bit TimeSeries
    //
    unsigned char* in = (unsigned char*)onebit_ts;
    
    for( unsigned ichan=0; ichan<n_freq; ichan++){
      float* out = output->get_datptr(ichan,0);

      unsigned char* from = in + ichan*ndat/8;

      unsigned i = 0;
      for( unsigned idat=0; idat<ndat; idat+=8, ++i)
	memcpy(out+idat,lookup+from[i],8*sizeof(float));
    }
    delete [] onebit_ts;
    
    rt2.stop();

    fprintf(stderr,"1: %f secs\t2: %f secs\n",rt1.get_total(),rt2.get_total());

  }  
  */
   
  if( verbose )
    fprintf(stderr,"Bye from OneBitCorrection\n");
}  

bool dsp::OneBitCorrection::matches (const Observation* observation)
{
  return observation->get_machine() == "PMDAQ" && observation->get_nbit() == 1;
}








