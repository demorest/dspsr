#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "dsp/Observation.h"
#include "dsp/TimeSeries.h"
#include "dsp/Bandpass.h"
#include "dsp/Response.h"
#include "dsp/Transformation.h"
#include "rfftw.h"
#include "genutil.h"
#include "assert.h"

bool dsp::Bandpass::verbose = false;

dsp::Bandpass::Bandpass ( const char * _name, Behaviour _type ) : Transformation<TimeSeries,Response> (_name, _type)
{
  nchan = 256;
}
void dsp::Bandpass::transformation ()
{
  // Number of points in fft
  int nsamp_fft = 0; 
  int npol = 0;

  npol = input->get_npol ();
  
  if (verbose)
    cerr << "dsp::Bandpass::npol is " << npol << endl;


    
  if (input->get_state() != Signal::Nyquist) {
    cerr << "Bandpass::transformation input data state may not be valid\n" << endl;
    return;
  }

  nsamp_fft = input->get_ndat ();
  nchan = nsamp_fft;
  
  float *scrap=0 ;
  scrap = new float [npol*2*nchan];
  
  register float *op = scrap;
  float *ptr_base = 0;
  register float *ptr=0;
  for (int ipol = 0; ipol < npol; ipol++) {
    
    if (verbose) 
      cerr << endl << "poln " << ipol << endl;
    
    ptr_base = (input->get_datptr(ipol,0)) ;
    
    if (verbose)
      cerr << "Pointer to data " << ptr_base << endl;

    for (int chan = 0; chan < nchan;chan++) {
      ptr = ptr_base + chan;
      if (verbose)
       cerr << "Incremented pointer" <<	ptr << " ..value " << *ptr << endl;
      
      *op = *ptr;
      op++;
      *op=0;
      op++;
      

      if (verbose) {
	fprintf(stderr,"Channel %d\r",chan);
	fflush(stderr);
      }
    }
  }	
  if (verbose)
    cerr << "dsp::Bandpass::built input data" << endl;
  // set the output
  // Need to pack the TimeSeries into a spectrum
  // this would work....but the spectrum would go out of scope....
  // Need to with define bandpass within calling function

  float * outp;
  outp = new float [2*nchan];
  
  if (verbose)
    cerr << "dsp::Bandpass::npoints in BP is " << nchan << endl;

  rfftw_plan p;
  p = rfftw_create_plan(nchan,FFTW_FORWARD,FFTW_ESTIMATE);

  if (verbose)
    cerr << "allocated scrap buffer" << endl;

  output->resize (npol, 1, nchan,1);
  if (verbose)
    cerr << "allocated output buffer" << endl;


  for (int ipol=0;ipol<npol;ipol++) {

    if (verbose)
      cerr << "FFT " << nchan << " points" << " polarisation " << ipol << endl;
    rfftw_one (p,&scrap[ipol*nchan*2],outp);
    if (verbose)
      cerr << "dsp::Bandpass::integrating (calling Response::integrate)" << endl;
    output->integrate (outp,ipol,0);
    if (verbose)
      cerr << "dsp::Bandpass::Finished integrating " << endl;
  }
  if (verbose)
    cerr << "dsp::Bandpass::deleting scrap buffer" << endl;

  delete [] scrap;
  delete [] outp;

  if (verbose)
    cerr << "dsp::Bandpass::operate::Finished" << endl;

}
dsp::Bandpass::~Bandpass () {
}

