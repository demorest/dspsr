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
  const uint32 mask = 0x01;

  const register unsigned nskip32 = n_freq/32;

  const unsigned algorithm = 8;

#define MM 512
#define NN 16

  //
  // Algorithm 1
  //
  if( algorithm==1 ){

    for (unsigned ichan=0; ichan < n_freq; ichan++) {
      unsigned shift32 = ichan % 32;
      
      const uint32* from = rawptr + ichan / 32;
      float* into = output -> get_datptr (ichan, 0);
      
      for (uint64 idat=0; idat < ndat; ++idat) {
	into[idat] = float (((*from)>>shift32) & mask);
	from += nskip32;
      }
    }
  }

  //
  // Algorithm 2 (Explicit float values)
  //

  else if( algorithm==2 ){
    for (unsigned ichan=0; ichan < n_freq; ichan++) {
      unsigned shift32 = ichan % 32;
      
      const uint32* from = rawptr + ichan / 32;
      float* into = output -> get_datptr (ichan, 0);
      
      for (uint64 idat=0; idat < ndat; ++idat) {
	if( (((*from)>>shift32) & mask)!=0 )
	  into[idat] = 1.0;
	else
	  into[idat] = 0.0;
	from += nskip32;
      }
    }
  }
  //
  // Algorithm 4 (Explicit float values with partially dynamic unwrap)
  //
  else if( algorithm==4 ){
    unsigned masks[NN];

    for( unsigned i=0; i<NN; i++)
      masks[i] = unsigned(pow(unsigned(2),unsigned(i)));

    for (unsigned ichan=0; ichan < n_freq; ichan+=NN) {
      register const unsigned shift = ichan % 32;
      
      const uint32* from = rawptr + ichan / 32;

      register float* intos[NN];
      for( unsigned i=0; i<NN; i++)
	intos[i] = output->get_datptr (ichan+i, 0);
      
      for (uint64 idat=0; idat < ndat; ++idat) {
	register const uint32 dat = (*from) >> shift;

	for( unsigned i=0; i<NN; i++)
	  if( dat & masks[i] ) intos[i][idat] = 1.0;
	  else	               intos[i][idat] = 0.0;

	from += nskip32;
      }
    }
  }
  //
  // Algorithm 8 (dynamic algorithm 7)
  //
  else if( algorithm==8 ){
    fprintf(stderr,"In alg 8\n");
      
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

    fprintf(stderr,"Leaving alg 8\n");
  }
  //
  // Algorithm 7 (alg 4 unravelled)
  //
  else if( algorithm==7 ){
    unsigned masks[16];

    for( unsigned i=0; i<16; i++)
      masks[i] = unsigned(pow(unsigned(2),unsigned(i)));

    const register unsigned nskip = n_freq/32;    

    for (unsigned ichan=0; ichan < n_freq; ichan+=16) {
      register const unsigned shift = ichan % 32;
      
      const uint32* from = rawptr + ichan / 32;

      register float* intos[16];
      for( unsigned i=0; i<16; i++)
	intos[i] = output->get_datptr (ichan+i, 0);
      
      for (uint64 idat=0; idat < ndat; idat+=16) {

	unsigned iindex = 0;

	register const uint32 dat = from[iindex] >> shift; iindex += nskip;
	register const uint32 dat2 = from[iindex] >> shift; iindex += nskip;
	register const uint32 dat3 = from[iindex] >> shift; iindex += nskip;
	register const uint32 dat4 = from[iindex] >> shift; iindex += nskip;
	register const uint32 dat5 = from[iindex] >> shift; iindex += nskip;
	register const uint32 dat6 = from[iindex] >> shift; iindex += nskip;
	register const uint32 dat7 = from[iindex] >> shift; iindex += nskip;
	register const uint32 dat8 = from[iindex] >> shift; iindex += nskip;
	register const uint32 dat9 = from[iindex] >> shift; iindex += nskip;
	register const uint32 dat10 = from[iindex] >> shift; iindex += nskip;
	register const uint32 dat11 = from[iindex] >> shift; iindex += nskip;
	register const uint32 dat12 = from[iindex] >> shift; iindex += nskip;
	register const uint32 dat13 = from[iindex] >> shift; iindex += nskip;
	register const uint32 dat14 = from[iindex] >> shift; iindex += nskip;
	register const uint32 dat15 = from[iindex] >> shift; iindex += nskip;
	register const uint32 dat16 = from[iindex] >> shift;

	for( unsigned i=0; i<NN; i++){
	  register float* fptr = intos[i]+idat;

	  if( dat & masks[i] )  fptr[0] = 1.0;
	  else	                fptr[0] = 0.0;
	  if( dat2 & masks[i] ) fptr[1] = 1.0;
	  else	                fptr[1] = 0.0;
	  if( dat3 & masks[i] ) fptr[2] = 1.0;
	  else	                fptr[2] = 0.0;
	  if( dat4 & masks[i] ) fptr[3] = 1.0;
	  else	                fptr[3] = 0.0;
	  if( dat5 & masks[i] ) fptr[4] = 1.0;
	  else	                fptr[4] = 0.0;
	  if( dat6 & masks[i] ) fptr[5] = 1.0;
	  else	                fptr[5] = 0.0;
	  if( dat7 & masks[i] ) fptr[6] = 1.0;
	  else	                fptr[6] = 0.0;
	  if( dat8 & masks[i] ) fptr[7] = 1.0;
	  else	                fptr[7] = 0.0;
	  if( dat9 & masks[i] ) fptr[8] = 1.0;
	  else	                fptr[8] = 0.0;
	  if( dat10 & masks[i] )fptr[9] = 1.0;
	  else	                fptr[9] = 0.0;
	  if( dat11 & masks[i] )fptr[10] = 1.0;
	  else	                fptr[10] = 0.0;
	  if( dat12 & masks[i] )fptr[11] = 1.0;
	  else	                fptr[11] = 0.0;
	  if( dat13 & masks[i] )fptr[12] = 1.0;
	  else	                fptr[12] = 0.0;
	  if( dat14 & masks[i] )fptr[13] = 1.0;
	  else	                fptr[13] = 0.0;
	  if( dat15 & masks[i] )fptr[14] = 1.0;
	  else	                fptr[14] = 0.0;
	  if( dat16 & masks[i] )fptr[15] = 1.0;
	  else	                fptr[15] = 0.0;
	}

	from += 16*nskip;
      }
    }
  }
  //
  // Algorithm 6 (No shift)
  //
  else if( algorithm==6 ){
    unsigned masks[32];

    for( unsigned i=0; i<32; i++)
      masks[i] = unsigned(pow(unsigned(2),unsigned(i)));

    for (unsigned ichan=0; ichan < n_freq; ichan+=32) {
      const uint32* from = rawptr + ichan / 32;

      float* intos[32];
      for( unsigned i=0; i<32; i++)
	intos[i] = output->get_datptr (ichan+i, 0);
      
      for (uint64 idat=0; idat < ndat; ++idat) {
	register const uint32 dat = *from;

	for( unsigned i=0; i<32; i++)
	  if( dat & masks[i] ) intos[i][idat] = 1.0;
	  else	               intos[i][idat] = 0.0;

	from += nskip32;
      }
    }
  }
  //
  // Algorithm 5 (Explicit float values with explicit unwrap)
  //
  else if( algorithm==5 ){
    register const uint32 mask2 = 2;
    register const uint32 mask3 = 4;
    register const uint32 mask4 = 8;
    register const uint32 mask5 = 16;
    register const uint32 mask6 = 32;
    register const uint32 mask7 = 64;
    register const uint32 mask8 = 128;
    register const uint32 mask9 = 256;
    register const uint32 mask10 = 512;
    register const uint32 mask11 = 1024;
    register const uint32 mask12 = 2048;
    register const uint32 mask13 = 4096;
    register const uint32 mask14 = 8192;
    register const uint32 mask15 = 16384;
    register const uint32 mask16 = 32768;

    for (unsigned ichan=0; ichan < n_freq; ichan+=NN) {
      unsigned shift = ichan % 32;
      
      const uint32* from = rawptr + ichan / 32;

      float* into0 = output->get_datptr (ichan, 0);
      float* into1 = output->get_datptr (ichan+1, 0);
      float* into2 = output->get_datptr (ichan+2, 0);
      float* into3 = output->get_datptr (ichan+3, 0);
      float* into4 = output->get_datptr (ichan+4, 0);
      float* into5 = output->get_datptr (ichan+5, 0);
      float* into6 = output->get_datptr (ichan+6, 0);
      float* into7 = output->get_datptr (ichan+7, 0);
      float* into8 = output->get_datptr (ichan+8, 0);
      float* into9 = output->get_datptr (ichan+9, 0);
      float* into10 = output->get_datptr (ichan+10, 0);
      float* into11 = output->get_datptr (ichan+11, 0);
      float* into12 = output->get_datptr (ichan+12, 0);
      float* into13 = output->get_datptr (ichan+13, 0);
      float* into14 = output->get_datptr (ichan+14, 0);
      float* into15 = output->get_datptr (ichan+15, 0);
     
      register uint32 dat;

      for (uint64 idat=0; idat < ndat; ++idat) {
	dat = (*from) >> shift;

	if( dat & mask )  into0[idat] = 1.0;
	else	          into0[idat] = 0.0;
	if( dat & mask2 ) into1[idat] = 1.0;
	else	          into1[idat] = 0.0;
	if( dat & mask3 ) into2[idat] = 1.0;
	else	          into2[idat] = 0.0;
	if( dat & mask4 ) into3[idat] = 1.0;
	else	          into3[idat] = 0.0;
	if( dat & mask5 ) into4[idat] = 1.0;
	else	          into4[idat] = 0.0;
	if( dat & mask6 ) into5[idat] = 1.0;
	else	          into5[idat] = 0.0;
	if( dat & mask7 ) into6[idat] = 1.0;
	else	          into6[idat] = 0.0;
	if( dat & mask8 ) into7[idat] = 1.0;
	else	          into7[idat] = 0.0;
	if( dat & mask9 )  into8[idat] = 1.0;
	else	          into8[idat] = 0.0;
	if( dat & mask10 ) into9[idat] = 1.0;
	else	          into9[idat] = 0.0;
	if( dat & mask11 ) into10[idat] = 1.0;
	else	          into10[idat] = 0.0;
	if( dat & mask12 ) into11[idat] = 1.0;
	else	          into11[idat] = 0.0;
	if( dat & mask13 ) into12[idat] = 1.0;
	else	          into12[idat] = 0.0;
	if( dat & mask14 ) into13[idat] = 1.0;
	else	          into13[idat] = 0.0;
	if( dat & mask15 ) into14[idat] = 1.0;
	else	          into14[idat] = 0.0;
	if( dat & mask16 ) into15[idat] = 1.0;
	else	          into15[idat] = 0.0;
	
	from += nskip32;
      }
    }
  }
  //
  // Algorithm 3 (Explicitly 8 channels at a time)
  //
  else if( algorithm==3 ){
     
    for( unsigned ichan=0; ichan<n_freq; ichan+=8){
      const uint32* from = rawptr + ichan/32;

      register float* into0 = output->get_datptr (ichan, 0);
      register float* into1 = output->get_datptr (ichan+1, 0);
      register float* into2 = output->get_datptr (ichan+2, 0);
      register float* into3 = output->get_datptr (ichan+3, 0);
      register float* into4 = output->get_datptr (ichan+4, 0);
      register float* into5 = output->get_datptr (ichan+5, 0);
      register float* into6 = output->get_datptr (ichan+6, 0);
      register float* into7 = output->get_datptr (ichan+7, 0);

      const register unsigned shift0 = ichan%32;
      
      register uint32 dat = 0;

      for (uint64 idat=0; idat < ndat; ++idat) {
	dat = *from;
	into0[idat] = float ((dat>>shift0) & mask);
	into1[idat] = float ((dat>>(shift0+1)) & mask);
	into2[idat] = float ((dat>>(shift0+2)) & mask);
	into3[idat] = float ((dat>>(shift0+3)) & mask);
	into4[idat] = float ((dat>>(shift0+4)) & mask);
	into5[idat] = float ((dat>>(shift0+5)) & mask);
	into6[idat] = float ((dat>>(shift0+6)) & mask);
	into7[idat] = float ((dat>>(shift0+7)) & mask);
	from += nskip32;
      }
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




