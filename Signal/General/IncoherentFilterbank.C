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
  //choice = 3;  /* the fastest by far */
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
  // Do the work (choice 1)
  //
  /*if( choice==1 ){

    if( verbose )
      fprintf(stderr,"dsp::IncoherentFilterbank::transformation(): doing choice 1\n");

    const size_t n_memcpy = nsamp_fft*sizeof(float);
    const unsigned stride = nsamp_fft;
    const unsigned channel_stride = output->get_subsize()*output->get_npol();

    auto_ptr<float> scratch(new float[nsamp_fft+2]);
    
    for( unsigned ipol=0;ipol<input->get_npol();ipol++){
      
      const float* in = input->get_datptr(0,ipol);
      float* detected = output->get_datptr(0,ipol);
      
      for( int ipart=0; ipart<npart; ++ipart){
	//fprintf(stderr,"ipol=%d ipart=%d\n",ipol,ipart);
	memcpy(scratch.get(),in+ipart*stride,n_memcpy);
	
	scfft1dc(scratch.get(), nsamp_fft, 1, wsave->begin()); 
	
	register const float* real = scratch.get()+1;
	register const float* imag = scratch.get()+nchan+1;
	
	for( unsigned ichan=0; ichan<nchan; ichan++) 
	  detected[ichan*channel_stride] = real[ichan]*real[ichan] + imag[ichan]*imag[ichan];
	
      }
      
    }
  }

  //
  // Do the work (choice 2)
  //
  else if( choice==2 ){

    if( verbose )
      fprintf(stderr,"dsp::IncoherentFilterbank::transformation(): doing choice 2\n");

    const unsigned stride = nsamp_fft;
    const unsigned channel_stride = output->get_subsize()*output->get_npol();

    for( unsigned ipol=0;ipol<input->get_npol();ipol++){
      
      float* in = input->get_datptr(0,ipol);
      float* detected = output->get_datptr(0,ipol);
      
      register float f1;
      register float f2;

      for( int ipart=0; ipart<npart; ++ipart){
	f1 = in[(ipart+1)*stride];
	f2 = in[(ipart+1)*stride+1];
	
	scfft1dc(in+ipart*stride, nsamp_fft, 1, wsave->begin()); 
	
	register const float* real = in+ipart*stride+1;
	register const float* imag = in+ipart*stride+nchan+1;
	
	for( unsigned ichan=0; ichan<nchan; ichan++) 
	  detected[ichan*channel_stride] = real[ichan]*real[ichan] + imag[ichan]*imag[ichan];
	in[(ipart+1)*stride] = f1;
	in[(ipart+1)*stride+1] = f2;
      }
      
    }
  }    
  */

  //
  // Do the work (choice 3) (destructive!)
  //
  //else if( choice==3 ){

    if( verbose )
      fprintf(stderr,"dsp::IncoherentFilterbank::transformation(): doing choice 3\n");

    const size_t n_memcpy = nsamp_fft*sizeof(float);
    const unsigned stride = nsamp_fft; 
    const unsigned npol= input->get_npol();

    auto_ptr<float> scratch(new float[nsamp_fft+2]);
    
    //RealTimer* loop1_timer = new RealTimer;
    //RealTimer* loop2_timer = new RealTimer;

    //RealTimer* fft_timer = new RealTimer;

    for( unsigned ipol=0;ipol<input->get_npol();ipol++){
      
      const float* in = input->get_datptr(0,ipol);
      
      //loop1_timer->start();
      for( int ipart=0; ipart<npart; ++ipart){

	// (1) memcpy to scratch	
	memcpy(scratch.get(),in+ipart*stride,n_memcpy);
	// (2) FFT	
	//fft_timer->start();
	scfft1dc(scratch.get(), nsamp_fft, 1, wsave->begin()); 
	//fft_timer->stop();

	register const float* real = scratch.get()+1;
	register const float* imag = scratch.get()+nchan+1;
	register float* det = (float*)in+ipart*nchan;
	// (3) SLD and add polarisations back
	for( unsigned i=0; i<nchan; i++)
	  det[i] = real[i]*real[i] + imag[i]*imag[i];
      }
      //loop1_timer->stop();

      float* out = output->get_datptr(0,ipol); 
      // (4) Convert the BitSeries to a TimeSeries in output's data array 

      //loop2_timer->start();
      for( unsigned ichan=0; ichan<nchan; ++ichan){
	register const float* from = in+ichan;
	register float* to = out+ichan*npol;
	
	for( int ipart=0; ipart<npart; ++ipart )
	  to[ipart] = from[ipart*nchan];
      }
      //loop2_timer->stop();

    }

    //fprintf(stderr,"loop 1 took          %f seconds\n",loop1_timer->get_total());
    //fprintf(stderr,"loop 2 took          %f seconds\n",loop2_timer->get_total());
    //fprintf(stderr,"Actual FFT'ing took  %f seconds\n",fft_timer->get_total());
    //}

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

