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

dsp::IncoherentFilterbank::IncoherentFilterbank () : Transformation<TimeSeries,TimeSeries> ("IncoherentFilterbank", outofplace){
  nchan = 0;
  state = Signal::Intensity;
  unroll_level = 16;
  floats_per_group = 0;

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

  vector<float> vdet;
  float* det = in0;

  if( !destroy_input ){
      vdet.resize(nsamp_fft*npart);
      det = &*vdet.begin();
  }

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
  det = get_input()->get_datptr(0,0);

  if( !destroy_input ){
      vdet.resize(nsamp_fft*npart);
      det = &*vdet.begin();
  }

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
  // Number of floats in the forward FFT
  const int nsamp_fft = nchan * 2/input->get_ndim();

  // Number of forward FFTs
  const int npart = input->get_ndat()/nsamp_fft;

  const size_t n_memcpy = nsamp_fft*sizeof(float);

  vector<float> vscratch(nsamp_fft+2);
  float* scratch = &*vscratch.begin();

  vector<float> vscratch2(nsamp_fft+2);
  float* scratch2 = &*vscratch2.begin();

  float* det = get_input()->get_datptr(0,0);

  vector<float> vdet;
  if( !destroy_input ){
      vdet.resize( nchan*npart );
      det = &*vdet.begin();
  }

  for( unsigned ipol=0; ipol<2; ipol++){  
    const float* in = input->get_datptr(0,ipol);

    fft_loop_timer.start();
    for( int ipart=0; ipart<npart; ++ipart, det+=nchan ){
      // (a) memcpy to scratch	
      memcpy(scratch,in+ipart*nsamp_fft,n_memcpy);
      // (b) FFT to scratch2
      fft_timer.start();
      fft::frc1d(nsamp_fft,scratch2,scratch);
      fft_timer.stop();
      // (d) SLD back
      for( unsigned i=0; i<nchan; ++i)
	  det[i] = SQR(scratch2[2*i]) + SQR(scratch2[2*i+1]);
    }
    fft_loop_timer.stop();

    // (d) Convert the BitSeries to a TimeSeries in output's data array 
    conversion_timer.start();
    for( unsigned ichan=0; ichan<nchan; ++ichan){
      register float* to = output->get_datptr(ichan,ipol);
      register const float* from = det+ichan;
      register unsigned i = 0;
      
      for( int ipart=0; ipart<npart; ++ipart, i += nchan )
	to[ipart] = from[i];
    }
    conversion_timer.stop();

  }    

}

void dsp::IncoherentFilterbank::form_undetected(){
  // Number of floats in the forward FFT
  const int nsamp_fft = nchan * 2/input->get_ndim();

  // Number of forward FFTs
  const unsigned npart = input->get_ndat()/nsamp_fft;

  vector<float> vscratch(nsamp_fft+2);
  float* scratch = &*vscratch.begin();

  vector<float> vscratch2(nsamp_fft*npart+2);
  float* scratch2 = &*vscratch.begin();

  for( unsigned ipol=0; ipol<2; ipol++){
    fft_loop_timer.start();
    float* in = input->get_datptr(0,ipol);

    for( unsigned ipart=0; ipart<npart; ++ipart){
      // (a) memcpy to scratch
      memcpy(scratch,in+ipart*nsamp_fft,nsamp_fft);

      float* scr = scratch2 + ipart*nsamp_fft;
      // (b) FFT to scratch2
      fft_timer.start();
      fft::frc1d(nsamp_fft,scr,scratch);
//      scfft1dc(scr, nsamp_fft, 1, wsave ); 
      fft_timer.stop();
    }
    fft_loop_timer.stop();

    // (3) Do the conversion back to a TimeSeries
    conversion_timer.start();

    unsigned fpg = floats_per_group;
    if( fpg == 0 )
	fpg = npart;

    const unsigned nfloat_fft = nsamp_fft*2;
    const unsigned total_floats = npart*2;

    // For each set of output channels...
    for( int ichan=0; ichan<nsamp_fft; ichan+=unroll_level){
      float** to = new float*[unroll_level];
      for( unsigned i=0; i<unroll_level; i++)
	to[i] = output->get_datptr(ichan+i,ipol);

      unsigned jfloat = 0;

      for( ; jfloat<total_floats; jfloat+=fpg){
	  const unsigned float_offset = jfloat*nsamp_fft + ichan;

	  for( unsigned ito=0; ito<unroll_level; ++ito){
	      unsigned counter = float_offset + ito;// jfloat*nsamp_fft + ichan+ito

	      for( unsigned ifloat=0; ifloat<fpg; ifloat+=2){
		  to[ito][jfloat+ifloat]   = scratch2[counter];
		  to[ito][jfloat+ifloat+1] = scratch2[counter+1];
		  counter += nfloat_fft;
	      }
	  }
      }

      if( jfloat < npart ) {
	  const unsigned float_offset = jfloat*nsamp_fft + ichan;
	  const unsigned floats_left = total_floats - jfloat;
	  
	  for( unsigned ito=0; ito<unroll_level; ++ito){
	      unsigned counter = float_offset + ito;// jfloat*nsamp_fft + ichan+ito
	      
	      for( unsigned ifloat=0; ifloat<floats_left; ifloat+=2){
		  to[ito][jfloat+ifloat]   = scratch2[counter];
		  to[ito][jfloat+ifloat+1] = scratch2[counter+1];
		  counter += nfloat_fft;
	      }
	  }
      }
      
      delete [] to;
    }
    conversion_timer.stop();

  }

}
