#include <memory>
#include <vector>

#include <stdio.h>
#include <stdlib.h>

#include "Error.h"
#include "Types.h"
#include "RealTimer.h"
#include "Reference.h"

#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"
#include "dsp/ChannelOrder.h"

#include "dsp/IncoherentFilterbank.h"

extern "C" { 
  void scfft1dc(float* data, int nfft, int isign, float* wsave);
}

dsp::IncoherentFilterbank::IncoherentFilterbank () : Transformation<TimeSeries,TimeSeries> ("IncoherentFilterbank", outofplace){
  nchan = 0;
  state = Signal::Intensity;
}

dsp::IncoherentFilterbank::~IncoherentFilterbank(){ }

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

  //
  // Set up the output
  //
  
  // Number of floats in the forward FFT
  int nsamp_fft = nchan * 2/input->get_ndim();

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
  output->set_rate( input->get_rate()/(nchan*2/input->get_ndim()));
  output->set_dc_centred( true );
  // should really change start time
  output->resize( npart );
 
  //
  // Set up wsave
  //
  if( !wsave.get() || wsave->size() != 2*nchan*2/input->get_ndim()+4 ){
    fprintf(stderr,"Acquiring plan\n");
    acquire_plan();
  }  

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
  const int npart = input->get_ndat()/nsamp_fft;

  const size_t n_memcpy = nsamp_fft*sizeof(float);
  const unsigned stride = nsamp_fft; 

  auto_ptr<float> scratch0(new float[nsamp_fft+2]);
  auto_ptr<float> scratch1(new float[nsamp_fft+2]);
  
  const float* in0 = input->get_datptr(0,0);
  const float* in1 = input->get_datptr(0,1);
  
  register float* det = (float*)in0;
  register const unsigned det_stride = nchan;

  for( int ipart=0; ipart<npart; ++ipart, det+=det_stride ){
    
    // (1) memcpy to scratch	
    memcpy(scratch0.get(),in0+ipart*stride,n_memcpy);
    memcpy(scratch1.get(),in1+ipart*stride,n_memcpy);
    
    // (2) FFT	
    scfft1dc(scratch0.get(), nsamp_fft, 1, wsave->begin()); 
    scfft1dc(scratch1.get(), nsamp_fft, 1, wsave->begin()); 
    
    // (3) SLD and add polarisations back
    register const float* real0 = scratch0.get();
    register const float* imag0 = real0 + nsamp_fft/2+1;
    
    register const float* real1 = scratch1.get();
    register const float* imag1 = real1 + nsamp_fft/2+1;
    
    for( unsigned i=0; i<nchan; ++i)
      det[i] = real0[i]*real0[i] + imag0[i]*imag0[i]  + real1[i]*real1[i] + imag1[i]*imag1[i];
  }
  
  // (4) Convert the BitSeries to a TimeSeries in output's data array 
  register float* to = output->get_datptr(0,0);
  register const unsigned to_stride = npart;
  register const unsigned from_stride = nchan;
  
  for( unsigned ichan=0; ichan<nchan; ++ichan, to += to_stride){
    register const float* from = in0+ichan;
    register unsigned i=0;
    
    for( int ipart=0; ipart<npart; ++ipart, i += from_stride )
      to[ipart] = from[i];
  }
  
}

void dsp::IncoherentFilterbank::form_PPQQ(){
  // Number of floats in the forward FFT
  const int nsamp_fft = nchan * 2/input->get_ndim();

  // Number of forward FFTs
  const int npart = input->get_ndat()/nsamp_fft;

  const size_t n_memcpy = nsamp_fft*sizeof(float);
  const unsigned stride = nsamp_fft; 

  auto_ptr<float> scratch(new float[nsamp_fft+2]);

  for( unsigned ipol=0; ipol<2; ipol++){  
    const float* in = input->get_datptr(0,ipol);
    
    register float* det = (float*)in;
    register const unsigned det_stride = nchan;
    
    for( int ipart=0; ipart<npart; ++ipart, det+=det_stride ){
      // (1) memcpy to scratch	
      memcpy(scratch.get(),in+ipart*stride,n_memcpy);
      // (2) FFT	
      scfft1dc(scratch.get(), nsamp_fft, 1, wsave->begin()); 
      // (3) SLD back
      register const float* real = scratch.get();
      register const float* imag = real + nsamp_fft+1;
      
      for( unsigned i=0; i<nchan; ++i)
	det[i] = real[i]*real[i] + imag[i]*imag[i];
    }
  
    // (4) Convert the BitSeries to a TimeSeries in output's data array 
    register float* to = output->get_datptr(0,ipol);
    register const unsigned to_stride = npart;
    register const unsigned from_stride = nchan;
    
    for( unsigned ichan=0; ichan<nchan; ++ichan, to += to_stride){
      register const float* from = in+ichan;
      register unsigned i = 0;
      
      for( int ipart=0; ipart<npart; ++ipart, i += from_stride )
	to[ipart] = from[i];
    }
  }    

}

void dsp::IncoherentFilterbank::form_undetected(){
  // Number of floats in the forward FFT
  const int nsamp_fft = nchan * 2/input->get_ndim();

  // Number of forward FFTs
  const int npart = input->get_ndat()/nsamp_fft;

  const size_t n_memcpy = nsamp_fft*sizeof(float);
  register const unsigned stride = nsamp_fft; 

  auto_ptr<float> scratch(new float[nsamp_fft+2]);

  for( unsigned ipol=0; ipol<2; ipol++){
    float* store = input->get_datptr(0,ipol);
    register float* scr = scratch.get();
  
    for( int ipart=0; ipart<npart; ++ipart, store+=stride ){
      // (1) memcpy to scratch	
      memcpy(scr,store,n_memcpy);
      // (2) FFT	
      scfft1dc(scr, nsamp_fft, 1, wsave->begin()); 
      // (3) copy back
      memcpy(store,scr,n_memcpy);
    }
  }

  /* (4) Convert to a TimeSeries */
  for( unsigned ipol=0; ipol<2; ipol++){
    float* from = input->get_datptr(0,ipol);
    float* to = output->get_datptr(0,ipol);

    register unsigned from_i = 0;
    register unsigned input_stride = nchan;
    register unsigned npt = 2*npart;

    for( unsigned ipt=0; ipt<npt; ++ipt, from_i += input_stride )
      to[ipt] = from[from_i];
  }

  /* old way- for if you were using scfft1d instead of scfft1dc
  // (4) Convert the BitSeries to a TimeSeries
  Reference::To<dsp::BitSeries> bs(new dsp::BitSeries);
  bs->Observation::operator=( *input );
  bs->set_state( Signal::Analytic );
  bs->set_ndim( 2 );
  bs->set_nchan( nchan );
  bs->set_npol( 2 );
  bs->rescale( nchan );
  bs->set_rate( input->get_rate()/(nchan*2/input->get_ndim()));
  bs->set_dc_centred( true );
  // should really change start time
  bs->attach( (unsigned char*)input->get_datptr(0,0) );

  Reference::To<dsp::ChannelOrder> order(new dsp::ChannelOrder);
  order->set_input( bs );
  order->set_output( output );
  order->set_rapid_variable( dsp::ChannelOrder::Channel );

  order->operate();

  output->attach( (float*)bs->get_rawptr() );*/
}

void dsp::IncoherentFilterbank::acquire_plan(){
  sink(wsave);
  auto_ptr<vector<float> > temp(new vector<float>(2*nchan*2/input->get_ndim()+4));
  wsave = temp;
  //fprintf(stderr,"wsave.get()=%p (*(wsave.get())).size()=%d, wsave->size()=%d\n",
  //  wsave.get(),(*(wsave.get())).size(),wsave->size());

  //fprintf(stderr,"Acquiring wsave with n=%d\n",2*nchan*2/input->get_ndim());
  scfft1dc(input->get_datptr(0,0),nchan*2/input->get_ndim(),0, wsave->begin());
  //fprintf(stderr,"wsave acquired\n");
}

