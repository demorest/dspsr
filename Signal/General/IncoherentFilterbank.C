#include <memory>
#include <vector>

#include <stdio.h>
#include <stdlib.h>

#include "Error.h"
#include "Types.h"
#include "RealTimer.h"

#include "dsp/TimeSeries.h"

#include "dsp/IncoherentFilterbank.h"

extern "C" { 
  void scfft1dc(float* data, int nfft, int isign, float* wsave);
}

dsp::IncoherentFilterbank::IncoherentFilterbank () : Transformation<TimeSeries,TimeSeries> ("IncoherentFilterbank", outofplace){
  nchan = 0;
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

  fprintf(stderr,"nchan=%d nsamp_fft=%d npart=%d\n",
	  nchan,nsamp_fft,npart);

  if( input->get_ndat()%nsamp_fft )
    fprintf(stderr,"dsp::IncoherentFilterbank::transformation() will throw away %d samples of the input and not FFT them.  (input->ndat ("UI64") doesn't divide nsamp_fft (%d)\n",
	    int(input->get_ndat())%nsamp_fft, input->get_ndat(), nsamp_fft);

  output->Observation::operator=(*input);
  output->set_state( Signal::Intensity );
  output->set_nchan( nchan );
  output->set_npol( 1 );
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
  const size_t n_memcpy = nsamp_fft*sizeof(float);
  const unsigned stride = nsamp_fft; 

  auto_ptr<float> scratch0(new float[nsamp_fft+2]);
  auto_ptr<float> scratch1(new float[nsamp_fft+2]);
  
  auto_ptr<float> scratch_big(new float[npart*nchan]);

  const float* in0 = input->get_datptr(0,0);
  const float* in1 = input->get_datptr(0,1);
  
  //register float* det = scratch_big.get();
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
    register const float* imag0 = scratch0.get()+1;
    
    register const float* real1 = scratch1.get();
    register const float* imag1 = scratch1.get()+1;  
    
    //register float* det = (float*)in0+ipart*nchan;
    //register float* det = scratch_big.get()+ipart*nchan;

    for( unsigned i=0; i<nchan; i+=2)
      det[i] = real0[i]*real0[i] + imag0[i]*imag0[i]  + real1[i]*real1[i] + imag1[i]*imag1[i];
  }

  // (4) Convert the BitSeries to a TimeSeries in output's data array 
  register float* to = output->get_datptr(0,0);
  register const unsigned to_stride = npart;
  register const unsigned from_stride = nchan;

  for( unsigned ichan=0; ichan<nchan; ++ichan, to += to_stride){
    register const float* from = in0+ichan;
    //register const float* from = scratch_big.get();
    register unsigned from_i = 0;
    
    for( int ipart=0; ipart<npart; ++ipart, from_i += from_stride )
      to[ipart] = from[from_i];
  }
    
  if( verbose )
    fprintf(stderr,"Returning from dsp::IncoherentFilterbank::transformation()\n");

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

