/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/WAPPUnpacker.h"
#include "dsp/WAPPFile.h"
#include "dsp/Scratch.h"

#include "machine_endian.h"
#include "FTransform.h"
#include "Error.h"

// from sigproc-2.4
#include "wapp_header.h"

#include <assert.h>

using namespace std;

// from sigproc.h
extern "C" {
  int vanvleck3lev(float *rho,int npts) ;
  void vanvleck9lev(float *rho,int npts) ;
  double inv_cerf(double input) ;
}

//! Constructor
dsp::WAPPUnpacker::WAPPUnpacker (const char* name) : HistUnpacker (name)
{
  set_ndig (4);
}

bool dsp::WAPPUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "WAPP";
}

/* time between correlator dumps in us */
#define WAPP_DEAD_TIME 0.34

void dsp::WAPPUnpacker::unpack ()
{
  const WAPPFile* wapp = get_Input<const WAPPFile>();

  if (!wapp)
    throw Error (InvalidState, "dsp::WAPPUnpacker::unpack",
		 "BitSeries::input is not a WAPPFile");

  assert (input->get_ndim() == 1);

  const uint64_t ndat = input->get_ndat();
  const unsigned npol = input->get_npol();
  const unsigned nbit = input->get_nbit();
  const unsigned nchan = input->get_nchan();
  const unsigned two_nchan = nchan * 2;

  if (verbose)
    cerr << "dsp::WAPPUnpacker::unpack ndat=" << ndat << " npol=" << npol
	 << " nbit=" << nbit << " nchan=" << nchan << endl;

  struct WAPP_HEADER* head = (struct WAPP_HEADER*) wapp->header;

  // /////////////////////////////////////////////////////////////////////
  //
  // calculate the scale factor to normalize correlation functions
  //
  double bw = input->get_bandwidth();
  if (bw < 0)
    bw = -bw;
  if (bw < 50.0) 
    bw = 50.0; /* correct scaling for narrow-band use */

  double tsamp_us = 1e6 / input->get_rate();

  /* correlator data rate */
  double crate = 1.0/(tsamp_us-WAPP_DEAD_TIME); 

  double scale = crate/bw;

  /* 9-level sampling */
  if (head->level==9)
    scale /= 16.0;

  /* summed IFs (search mode) */
  if (head->sum)
    scale /= 2.0;

  /* needed for truncation modes */
  scale *= pow(2.0,(double)head->lagtrunc);

  /* now define a number of working arrays to store lags and spectra */
  float* acf = scratch->space<float> (5*nchan);
  float* psd = acf + 2*nchan;
  float* window = psd + 2*nchan;

  bool hanning = true;
  bool hamming = false;

  /* set up the weights for windowing of ACF to monimize FFT leakage */
  /* no window (default) */
  double hweight = 1.0;
  if (hanning) {
    /* Hanning window */
    hweight=0.50;
  } else if (hamming) {
    /* Hamming window */
    hweight=0.54;
  }

  /* define the smoothing window to be applied base on the above weight */
  for (unsigned ichan=0; ichan<nchan; ichan++)
    window[ichan]=(hweight+(1.0-hweight)*cos((M_PI*ichan)/nchan));

  const unsigned char* from8 = input->get_rawptr();
  uint16_t* from16 = (uint16_t*) input->get_rawptr();
  uint32_t* from32 = (uint32_t*) input->get_rawptr();

  bool do_vanvleck = true;

  for (unsigned idat=0; idat<ndat; idat++) {

    for (unsigned ipol=0; ipol<npol; ipol++) {

      unsigned ichan = 0;

      /* fill lag array with scaled CFs */
      switch (nbit) {
      case 8:
	for (ichan=0; ichan<nchan; ichan++)
	  acf[ichan] = scale * double(from8[ichan]) - 1.0;
	from8 += nchan;
	break;
      case 16:
	for (ichan=0; ichan<nchan; ichan++) {
          FromLittleEndian(from16[ichan]);
	  acf[ichan] = scale * double(from16[ichan]) - 1.0;
        }
	from16 += nchan;
	break;
      case 32:
	for (ichan=0; ichan<nchan; ichan++) {
          FromLittleEndian(from32[ichan]);
	  acf[ichan] = scale * double(from32[ichan]) - 1.0;
        }
	from32 += nchan;
	break;
      }

      /* calculate power and correct for finite level quantization */
      double power = inv_cerf(acf[0]);
      power = 0.1872721836/power/power;
      if (ipol<2) {
	if (do_vanvleck) { 
	  if (head->level==1) {
	    /* apply standard 3-level van vleck correction */
	    vanvleck3lev(acf,nchan);
	  } else if (head->level==2) {
	    /* apply 9-level van vleck correction */
	    vanvleck9lev(acf,nchan);
	  }
	}
      }
      
      /* form windowed even ACF in array */
      for (ichan=1; ichan<nchan; ichan++) {
	acf[ichan]=window[ichan]*acf[ichan]*power;
	acf[two_nchan-ichan]=acf[ichan];
      }   

      acf[nchan]=0.0;
      acf[0]=acf[0]*power; 

      /* FFT the ACF (which is real and even) -> real and even FFT */
      FTransform::frc1d (two_nchan,psd,acf);

      for (ichan=0; ichan<nchan; ichan++)
	output->get_datptr(ichan, ipol)[idat] = psd[ichan*2];

    } /* for each polarization (IF) */

  } /* for each time slice */

}


