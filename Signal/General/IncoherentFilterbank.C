#include <memory>
#include <vector>

#include <stdio.h>
#include <stdlib.h>

#include "Error.h"
#include "Types.h"
#include "RealTimer.h"
#include "Reference.h"
#include "fftm.h"

#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"
#include "dsp/ChannelOrder.h"

#include "dsp/IncoherentFilterbank.h"

#ifdef ACTIVATE_MKL
extern "C" { 
  void scfft1dc(float* data, int nfft, int isign, float* wsave);
}
#endif

dsp::IncoherentFilterbank::IncoherentFilterbank () : Transformation<TimeSeries,TimeSeries> ("IncoherentFilterbank", outofplace){
  nchan = 0;
  state = Signal::Intensity;
  unroll_level = 16;

  wsave = 0;
  wsave_size = 0;
  destroy_input = false;
}

dsp::IncoherentFilterbank::~IncoherentFilterbank(){ free_plan(); }

void dsp::IncoherentFilterbank::transformation(){
  //
  // Initial error checking
  //
  if( nchan<2 )
    throw Error (InvalidState, "dsp::IncoherentFilterbank::transformation",
		 "invalid number of channels = %d", nchan);

  if( !power_of_two(nchan) )
    throw Error (InvalidState, "dsp::IncoherentFilterbank::transformation",
		 "nchan (%d) isn't a power of two", nchan);

  if( input->get_nchan()!=1 )
    throw Error (InvalidState, "dsp::IncoherentFilterbank::transformation",
		 "Sorry, but this class isn't set up to form filterbanks from filterbanks.  Why don't you fix it?");

  if( input->get_state()!=Signal::Nyquist )
    throw Error (InvalidState, "dsp::IncoherentFilterbank::transformation",
		 "Your input state isn't Signal::Nyquist.  This routine is only written for raw CPSR-II data, which should be Signal::Nyquist\n");

  if( input->get_npol()!=2 )
    throw Error (InvalidState, "dsp::IncoherentFilterbank::transformation",
		 "Your input doesn't have 2 polarisations.  This routine is written only for raw CPSR-II data, which should have 2 polarisations.");

  if( state==Signal::Analytic && unroll_level>nchan ){
    fprintf(stderr,"dsp::IncoherentFilterbank::transformation: your unroll level (%d) was greater than your nchan (%d).  It has been reduced to %d\n",
	    unroll_level,nchan,nchan);
    unroll_level = nchan;
  }

  //
  // Set up the output
  //
  
  // The multiplicative factor in going real->complex
  int real2complex = 2/input->get_ndim();

  // Number of floats in the forward FFT
  int nsamp_fft = nchan * real2complex;

  // Number of forward FFTs
  int npart = input->get_ndat()/nsamp_fft;

  // Number of output polarisations
  int output_npol, output_ndim;
  if( state==Signal::Intensity )
    { output_npol = 1; output_ndim = 1; }
  else if( state==Signal::PPQQ )
    { output_npol = 2; output_ndim = 1; }
  else if( state==Signal::Analytic )
    { output_npol = 2; output_ndim = 2; }
  else
    throw Error(InvalidState,"dsp::IncoherentFilterbank::transformation()",
		"You need to call dsp::IncoherentFilterbank::set_state() with an argument of: Signal::Intensity, Signal::PPQQ or Signal::Analytic");

  if( verbose )
    fprintf(stderr,"nchan=%d nsamp_fft=%d npart=%d\n",
	    nchan,nsamp_fft,npart);

  if( input->get_ndat()%nsamp_fft )
    fprintf(stderr,"dsp::IncoherentFilterbank::transformation() will throw away %d samples of the input and not FFT them.  (input->ndat ("UI64") doesn't divide nsamp_fft (%d)\n",
	    int(input->get_ndat())%nsamp_fft, input->get_ndat(), nsamp_fft);

  output->Observation::operator=(*input);
  output->set_state( state );
  output->set_ndim( output_ndim );
  output->set_nchan( nchan );
  output->set_npol( output_npol );
  output->rescale( nchan );
  output->set_rate( input->get_rate()/(nchan*real2complex));
  // First channel corresponds to DC which is centred at the edge of the band
  output->set_dc_centred( true );

  if( verbose )
    fprintf(stderr,"Calling output->resize(%d) with nchan=%d npol=%d ndim=%d\n",
	    npart,output->get_nchan(),output->get_npol(),output->get_ndim());
  output->resize( npart );

  // Set up wsave 
  acquire_plan();

  //
  // Do the work (destroys input array!)
  //
  if( state==Signal::Intensity )
    form_stokesI();
  else if( state==Signal::PPQQ )
    form_PPQQ();
  else if( state==Signal::Analytic )
    form_undetected();
 
  if( verbose )
    fprintf(stderr,"Returning from dsp::IncoherentFilterbank::transformation()\n");
}
  
void dsp::IncoherentFilterbank::form_stokesI(){
  // Number of floats in the forward FFT
  const int nsamp_fft = nchan * 2/input->get_ndim();

  // Number of forward FFTs
  const int npart = get_input()->get_ndat()/nsamp_fft;

  const size_t n_memcpy = nsamp_fft*sizeof(float);

  float* in0 = get_input()->get_datptr(0,0);
  float* in1 = get_input()->get_datptr(0,1);

  vector<float> vto0(nsamp_fft+2);
  float* to0 = &*vto0.begin();

  vector<float> vto1(nsamp_fft+2);
  float* to1 = &*vto1.begin();

  vector<float> vfrom(nsamp_fft+2);
  float* from = &*vfrom.begin();

  vector<float> vdet(nsamp_fft*npart);
  float* det = &*vdet.begin();

  fft_loop_timer.start();
  for( int ipart=0; ipart<npart; ++ipart){
      // (a) get pol0 input ready to transform
      memcpy( from, in0, n_memcpy);
      
      // (b) Transform pol0 into scratch0
      fft_timer.start();
      fft::frc1d( nsamp_fft, to0, from);
      fft_timer.stop();

      // (c) get pol1 input ready to transform
      memcpy( from, in1, n_memcpy);

      // (d) Transform pol1 into scratch1
      fft_timer.start();
      fft::frc1d( nsamp_fft, to1, from);
      fft_timer.stop();
  
      // (e) SLD
      for( int i=0; i<nsamp_fft; ++i)
	  det[i] = SQR(to0[2*i]) + SQR(to0[2*i+1]) + SQR(to1[2*i]) + SQR(to1[2*i+1]);

      det += nsamp_fft;
      in0 += nsamp_fft;
      in1 += nsamp_fft;
  }
  fft_loop_timer.stop();

  // (e) Convert the BitSeries to a TimeSeries in output's data array 
  det = &*vdet.begin();

  conversion_timer.start();
  for( unsigned ichan=0; ichan<nchan; ++ichan){
    float* to = output->get_datptr(ichan,0);
    const float* from = det + ichan;

    unsigned i=0;

    for( int ipart=0; ipart<npart; ++ipart, i += nchan )
	to[ipart] = from[i];
  }
  conversion_timer.stop();
  
}

void dsp::IncoherentFilterbank::form_PPQQ(){
  if( verbose ) fprintf(stderr,"In form_PPQQ()\n");

#ifndef ACTIVATE_MKL
  throw Error(InvalidState,"dsp::IncoherentFilterbank::form_PPQQ()",
	      "MKL not activated");
#else
  // Number of floats in the forward FFT
  const int nsamp_fft = nchan * 2/input->get_ndim();

  // Number of forward FFTs
  const int npart = input->get_ndat()/nsamp_fft;

  const size_t n_memcpy = nsamp_fft*sizeof(float);

  auto_ptr<float> scratch(new float[nsamp_fft+2]);

  for( unsigned ipol=0; ipol<2; ipol++){  
    const float* in = input->get_datptr(0,ipol);

    float* big_scratch = (float*)input->get_datptr(0,ipol);
    if( !destroy_input )
      big_scratch = new float[nchan*npart]; 
    
    register float* det = big_scratch;

    fft_loop_timer.start();
    for( int ipart=0; ipart<npart; ++ipart, det+=nchan ){
      // (1) memcpy to scratch	
      memcpy(scratch.get(),in+ipart*nsamp_fft,n_memcpy);
      // (2) FFT	
      fft_timer.start();
      scfft1dc(scratch.get(), nsamp_fft, 1, wsave ); 
      fft_timer.stop();
      // (3) SLD back
      // MKL packs output as DC, r1, r2, ... , r(n/2), 0 , i1, i2, ... , i(n/2-1), 0
      register const float* real = scratch.get();
      register const float* imag = real + nsamp_fft/2+1;
      
      for( unsigned i=0; i<nchan; ++i)
	det[i] = real[i]*real[i] + imag[i]*imag[i];
    }
    fft_loop_timer.stop();

    // (4) Convert the BitSeries to a TimeSeries in output's data array 
    conversion_timer.start();
    for( unsigned ichan=0; ichan<nchan; ++ichan){
      register float* to = output->get_datptr(ichan,ipol);
      register const float* from = big_scratch+ichan;
      register unsigned i = 0;
      
      for( int ipart=0; ipart<npart; ++ipart, i += nchan )
	to[ipart] = from[i];
    }
    conversion_timer.stop();

    if( !destroy_input )
      delete [] big_scratch;
  }    

  if( verbose ) fprintf(stderr,"Returning from form_PPQQ()\n");
#endif
}

void dsp::IncoherentFilterbank::form_undetected(){
#ifndef ACTIVATE_MKL
  throw Error(InvalidState,"dsp::IncoherentFilterbank::form_undetected()",
	      "MKL not activated");
#else
  if( verbose ) fprintf(stderr,"In form_undetected()\n");

  // Number of floats in the forward FFT
  const int nsamp_fft = nchan * 2/input->get_ndim();

  // Number of forward FFTs
  const unsigned npart = input->get_ndat()/nsamp_fft;

  // Number of floats out per forward FFT
  // ( MKL packs output as DC, r1, r2, ... , r(n/2), 0 , i1, i2, ... , i(n/2-1), 0 )
  register const int floats_out = nsamp_fft+2;

  auto_ptr<float> scratch(new float[floats_out*npart]);

  for( unsigned ipol=0; ipol<2; ipol++){

    fft_loop_timer.start();
    float* in = input->get_datptr(0,ipol);

    for( unsigned ipart=0; ipart<npart; ++ipart){
      // (1) memcpy to scratch
      register float* scr = scratch.get() +ipart*floats_out;
      memcpy(scr,in+ipart*nsamp_fft,nsamp_fft);
      // (2) FFT	
      fft_timer.start();
      scfft1dc(scr, nsamp_fft, 1, wsave ); 
      fft_timer.stop();
    }
    fft_loop_timer.stop();

    // (3) Do the conversion back to a TimeSeries
    conversion_timer.start();

    // For each set of output channels...
    for( unsigned ichan=0; ichan<output->get_nchan(); ichan+=unroll_level){
      register float* from_re = scratch.get();
      
      float** to = new float*[unroll_level];
      for( unsigned i=0; i<unroll_level; i++)
	to[i] = output->get_datptr(ichan+i,ipol);
      
      register unsigned from_i = 0;
      register const unsigned imag_offset = nsamp_fft/2+1;
      register const unsigned nfloats = npart*2;

      // For each set of complex values out...
      for( unsigned i=0; i<nfloats; i+=2, from_i += floats_out ){
	// For each output channel in the group copy the imaginary and real bits over...
	for( unsigned ito=0; ito<unroll_level; ++ito){
	  to[ito][i] = from_re[from_i+ito];
	  to[ito][i+1] = from_re[from_i+ito+imag_offset];
	}
      }

      delete [] to;
    }
    conversion_timer.stop();

  }

  if( verbose ) fprintf(stderr,"Returning from form_undetected()\n");
#endif
}

void dsp::IncoherentFilterbank::acquire_plan(){
#ifndef ACTIVATE_MKL
  throw Error(InvalidState,"dsp::IncoherentFilterbank::acquire_plan()",
	      "MKL not activated");
#else
  int real2complex = 2/input->get_ndim();
  
  // The size of wsave required by MKL is 2n+4
  uint64 magic_size = 2*nchan*real2complex + 4;

  if( wsave && wsave_size == magic_size ){
    // wsave is already the correct size!
    return;
  }

  free_plan();
  wsave = new float[magic_size];
  wsave_size = magic_size;

  if( verbose )
    fprintf(stderr,"Have allocated "UI64" floats for wsave=%p and n=%d\n",
	    magic_size,wsave,nchan*real2complex);

  scfft1dc(input->get_datptr(0,0),nchan*real2complex,0,wsave );
  if( verbose )
    fprintf(stderr,"wsave acquired\n");
#endif
}





