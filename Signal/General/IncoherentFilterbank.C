/***************************************************************************
 *
 *   Copyright (C) 2003 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/IncoherentFilterbank.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"

#include "Error.h"
#include "FTransform.h"
#include "templates.h"

#include <string.h>

using namespace std;

dsp::IncoherentFilterbank::IncoherentFilterbank ()
  : Transformation<TimeSeries,TimeSeries> ("IncoherentFilterbank", outofplace)
{
  nchan = 0;
  state = Signal::Intensity;
  unroll_level = 16;

  wsave = 0;
  wsave_size = 0;
  destroy_input = false;
}

dsp::IncoherentFilterbank::~IncoherentFilterbank()
{
  free_plan(); 
}

uint64_t power_of_two (uint64_t number)
{
  uint64_t twos = 1;
  while (twos < number)
    twos *= 2;
  if (twos != number)
    return 0;
  return 1;
}

void dsp::IncoherentFilterbank::transformation()
{
  //
  // Initial error checking
  //
  if( nchan < 2*input->get_nchan() )
    throw Error (InvalidState, "dsp::IncoherentFilterbank::transformation",
		 "Output nchan (%d) should be at least twice the number of input channels (%d)",
		 nchan,input->get_nchan());

  if( nchan % input->get_nchan() != 0 )
    throw Error (InvalidState, "dsp::IncoherentFilterbank::transformation",
		 "nchan out (%d) doesn't divide the number of input channels (%d)",
		 nchan, input->get_nchan());

  if( !power_of_two(nchan/input->get_nchan()) )
    throw Error (InvalidState, "dsp::IncoherentFilterbank::transformation",
		 "nchan/input->nchan (%d/%d) isn't a power of two", 
		 nchan,input->get_nchan());

  if( input->get_state()!=Signal::Nyquist )
    throw Error (InvalidState, "dsp::IncoherentFilterbank::transformation",
		 "Your input state isn't Signal::Nyquist.  This routine is only written for raw CPSR-II data, which should be Signal::Nyquist\n");

  if( input->get_npol()!=2 )
    throw Error (InvalidState, "dsp::IncoherentFilterbank::transformation",
		 "Your input doesn't have 2 polarisations.  This routine is written only for raw CPSR-II data, which should have 2 polarisations.");

  if( state==Signal::Analytic && unroll_level>nchan/input->get_nchan() ){
    fprintf(stderr,"dsp::IncoherentFilterbank::transformation: your unroll level (%d) was greater than your nchan out per input channel (%d).  It has been reduced\n",
	    unroll_level,nchan/input->get_nchan());
    unroll_level = nchan/input->get_nchan();
  }

  //
  // Set up the output
  //
  
  //! Number of channels outputted per input channel
  unsigned nchan_subband = nchan / input->get_nchan();

  // The multiplicative factor in going real->complex
  int real2complex = 2/input->get_ndim();

  // Number of forward FFTs
  int npart = (input->get_ndat()*input->get_ndim()) / (2*nchan_subband);

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

  get_output()->copy_configuration( get_input() );
  get_output()->set_state( state );
  get_output()->set_ndim( output_ndim );
  get_output()->set_nchan( nchan );
  get_output()->set_npol( output_npol );
  
  output->rescale( nchan_subband );
  output->set_rate( input->get_rate()/(nchan_subband*real2complex));
  // First channel corresponds to DC which is centred at the edge of the band
  output->set_dc_centred( true );

  if( verbose )
    fprintf(stderr,"Calling output->resize(%d) with nchan=%d npol=%d ndim=%d\n",
	    npart,output->get_nchan(),output->get_npol(),output->get_ndim());
  output->resize( npart );

  //
  // Do the work
  //
  if( state==Signal::Intensity )
    form_stokesI();
  else if( state==Signal::PPQQ )
    form_PPQQ();
  else if( state==Signal::Analytic )
    form_undetected();
  else
    throw Error(InvalidState,"dsp::IncoherentFilterbank()",
		"state '%s' not recognised",State2string(state).c_str());
}

typedef void (*fft_call) (size_t nfft, float* into, const float* from);

void dsp::IncoherentFilterbank::form_stokesI(){
  //! Number of channels outputted per input channel
  unsigned nchan_subband = nchan / input->get_nchan();

  const int npart = (input->get_ndat()*input->get_ndim()) / (2*nchan_subband);

  vector<float> time0(nchan_subband*2+2);
  vector<float> time1(nchan_subband*2+2);
  vector<float> sp0(nchan_subband*2+2); // sp = spectra
  vector<float> sp1(nchan_subband*2+2);
  vector<float> detected(npart*nchan_subband);

  for( unsigned i_input_chan=0; i_input_chan<input->get_nchan(); i_input_chan++){
  
    const float* in0 = input->get_datptr(i_input_chan,0);
    const float* in1 = input->get_datptr(i_input_chan,1);

    float* det = &*detected.begin();

    fft_call forwards = FTransform::frc1d;
    if( input->get_ndim() == 2 )  forwards = FTransform::fcc1d;
    unsigned fft_sz = nchan_subband;
    if( input->get_ndim() == 1 )  fft_sz *= 2;

    fft_loop_timer.start();    
    for( int ipart=0; ipart<npart; ++ipart, det+=nchan_subband ){
      // (1) memcpy to scratch	
      memcpy( &*time0.begin(), in0+ipart*nchan_subband*2, nchan_subband*2*sizeof(float));
      memcpy( &*time1.begin(), in1+ipart*nchan_subband*2, nchan_subband*2*sizeof(float));    

      // (2) FFT	
      fft_timer.start();
      forwards( fft_sz, &*sp0.begin(), &*time0.begin());
      forwards( fft_sz, &*sp1.begin(), &*time1.begin());
      fft_timer.stop();

      // (3) SLD and add polarisations back
      for( unsigned i=0; i<nchan_subband; i++)
	det[i] = sqr(sp0[2*i]) + sqr(sp0[2*i+1]) + sqr(sp1[2*i]) + sqr(sp1[2*i+1]);
    }
    fft_loop_timer.stop();

    // (4) Convert the BitSeries to a TimeSeries in output's data array 
    conversion_timer.start();
    for( unsigned ichan=0; ichan<nchan_subband; ++ichan){
      register float* to = output->get_datptr(i_input_chan*nchan_subband+ichan,0);
      register const float* from = &*detected.begin()+ichan;
      register unsigned i=0;
      
      for( int ipart=0; ipart<npart; ++ipart, i += nchan_subband )
	to[ipart] = from[i];
    }
    conversion_timer.stop();
  }

}

void dsp::IncoherentFilterbank::form_PPQQ(){
  //! Number of channels outputted per input channel
  unsigned nchan_subband = nchan / input->get_nchan();

  const int npart = (input->get_ndat()*input->get_ndim()) / (2*nchan_subband);

  vector<float> time0(nchan_subband*2+2);
  vector<float> sp0(nchan_subband*2+2); // sp = spectra
  vector<float> detected(npart*nchan_subband);

  fft_call forwards = FTransform::frc1d;
  if( input->get_ndim() == 2 )  forwards = FTransform::fcc1d;
  unsigned fft_sz = nchan_subband;
  if( input->get_ndim() == 1 )  fft_sz *= 2;

  for( unsigned i_input_chan=0; i_input_chan<input->get_nchan(); i_input_chan++){
    
    fft_loop_timer.start();
    for( unsigned ipol=0; ipol<input->get_npol(); ipol++){
      const float* in0 = input->get_datptr(i_input_chan,ipol);
      float* det = &*detected.begin();

      for( int ipart=0; ipart<npart; ++ipart, det+=nchan_subband ){
	// (1) memcpy to scratch	
	memcpy( &*time0.begin(), in0+ipart*nchan_subband*2, nchan_subband*2*sizeof(float));

	// (2) FFT	
	fft_timer.start();
	forwards( fft_sz, &*sp0.begin(), &*time0.begin());
	fft_timer.stop();

	// (3) SLD
	for( unsigned i=0; i<nchan_subband; i++)
	  det[i] = sqr(sp0[2*i]) + sqr(sp0[2*i+1]);
      }
      fft_loop_timer.stop();

      // (4) Convert the BitSeries to a TimeSeries in output's data array 
      conversion_timer.start();
      for( unsigned ichan=0; ichan<nchan_subband; ++ichan){
	register float* to = output->get_datptr(i_input_chan*nchan_subband+ichan,ipol);
	register const float* from = &*detected.begin()+ichan;
	register unsigned i=0;
	
	for( int ipart=0; ipart<npart; ++ipart, i += nchan_subband )
	  to[ipart] = from[i];
      }
      conversion_timer.stop();    
    }
  }

}

void dsp::IncoherentFilterbank::form_undetected(){
  //! Number of channels outputted per input channel
  unsigned nchan_subband = nchan / input->get_nchan();

  const int npart = (input->get_ndat()*input->get_ndim()) / (2*nchan_subband);

  vector<float> time0(nchan_subband*2+2);
  vector<float> sp0(nchan_subband*2+2); // sp = spectra
  vector<float> undetected(npart*nchan_subband*2);
  
  fft_call forwards = FTransform::frc1d;
  if( input->get_ndim() == 2 )  forwards = FTransform::fcc1d;
  unsigned fft_sz = nchan_subband;
  if( input->get_ndim() == 1 )  fft_sz *= 2;
  
  for( unsigned i_input_chan=0; i_input_chan<input->get_nchan(); i_input_chan++){
    
    fft_loop_timer.start();
    for( unsigned ipol=0; ipol<input->get_npol(); ipol++){
      const float* in0 = input->get_datptr(i_input_chan,ipol);
      float* undet = &*undetected.begin();
      
      for( int ipart=0; ipart<npart; ++ipart, undet+=nchan_subband*2 ){
	// (1) memcpy to scratch	
	memcpy( &*time0.begin(), in0+ipart*nchan_subband*2, nchan_subband*2*sizeof(float));

	// (2) FFT	
	fft_timer.start();
	forwards( fft_sz, &*sp0.begin(), &*time0.begin());
	fft_timer.stop();
	
	// (3) Copy into 'undetected'
	memcpy( undet, &*sp0.begin(), nchan_subband*2*sizeof(float));
      }
      fft_loop_timer.stop();
      
      int method = 2;
      if( method==1 ){
	// (4) Convert the BitSeries to a TimeSeries in output's data array 
	conversion_timer.start();
	for( unsigned ichan=0; ichan<nchan_subband; ++ichan){
	  register float* to = output->get_datptr(i_input_chan*nchan_subband*ichan,ipol);
	  register const float* from = &*undetected.begin()+ichan*2;
	  register unsigned i=0;
	  
	  for( int ipart=0; ipart<npart; ipart++, i += nchan_subband ){
	    to[2*ipart]   = from[2*i];
	    to[2*ipart+1] = from[2*i+1];
	  }
	}
	conversion_timer.stop();    
      }
      else{

	// (4) Convert the BitSeries to a TimeSeries in output's data array 
	// For each set of output channels...
	for( unsigned ichan=0; ichan<nchan_subband; ichan+=unroll_level){
	  register float* from = &*undetected.begin() + ichan*2;
    
	  vector<float*> to(unroll_level);
	  for( unsigned i=0; i<unroll_level; i++)
	    to[i] = output->get_datptr(i_input_chan*nchan_subband+ichan+i,ipol);
    
	  register unsigned from_i = 0;
	  
	  // For each set of complex values out...
	  for( int ipart=0; ipart<npart; ipart++, from_i += nchan_subband*2 ){
	    // For each output channel in the group copy the imaginary and real bits over...
	    for( unsigned ito=0; ito<unroll_level; ++ito){
	      to[ito][2*ipart]   = from[from_i+2*ito];
	      to[ito][2*ipart+1] = from[from_i+2*ito+1];
	    }
	  }  
	}
	conversion_timer.stop();
      }
    }
  
  } //foreach i_input_chan

}
  
