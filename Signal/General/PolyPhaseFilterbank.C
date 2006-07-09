/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
//  FilterBank.cc:
//
//  Copyright (C) 2002
//  ASTRON (Netherlands Foundation for Research in Astronomy)
//  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//

#include <stdio.h>             // for sprintf
#include <fstream.h>
#include "PolyPhaseFilterbank.h"

template<class Type>
FilterBank<Type>::FilterBank(string CoefficientFile, int OverlapSamples, int Real)
{
  // Open a stream to the file where the filter coefficients are located
  ifstream CoeffFile(CoefficientFile.c_str(), ifstream::in);

  // Read the filter coefficients and place them in a matrix
  CoeffFile >> itsFilterCoefficients;

   // Initialize the member variables
  itsNumberOfBands  = itsFilterCoefficients.rows();
  itsOrder          = itsFilterCoefficients.cols();
  itsMatrixPosition = 0;
  itsOverlapSamples = OverlapSamples;
  isReal            = Real;
  itsReArrangedSignal.resize(itsNumberOfBands, itsOrder);

  // Initialize FFTW
  fftplancomplex = fftw_create_plan(itsNumberOfBands, FFTW_FORWARD, FFTW_ESTIMATE);
  fftplanreal = rfftwnd_create_plan(1, &itsNumberOfBands, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);
}


template<class Type>
FilterBank<Type>::~FilterBank()
{
  fftw_destroy_plan(fftplancomplex);
  rfftwnd_destroy_plan(fftplanreal);
}


template<class Type>
blitz::Array<complex<float>, 2> FilterBank<Type>::filter(blitz::Array<Type, 1> Input)
{
  blitz::Array<complex<float>, 2> FilterBankOutput(itsNumberOfBands, 1);
  blitz::Array<complex<float>, 1> Temp(itsNumberOfBands / 2);

  if (itsOverlapSamples > 0)
  {
    // Rearrange the input so that it can be easily convolved with the filtercoefficients
    ReArrangeWithOverlap(Input);
  }
  else
  {
    ReArrange(Input);
  }

  // Convolve the rearrangedsignals with the filtercoefficients
  blitz::Array<Type, 1> ConvolvedSignal = Convolve();

  // Do a FFT n the convolved signal and return the output
  FilterBankOutput = FFT(ConvolvedSignal);

  return FilterBankOutput;
}


template<class Type>
void FilterBank<Type>::ReArrange(blitz::Array<Type, 1> Input) // This function might be not necessary instead use ReArrange with overlap with overlap = 0
{
  // The input must be number of bands long!
  for (int i = 0; i < itsNumberOfBands; ++i)
  {
    itsReArrangedSignal(i, itsMatrixPosition) = Input(i);
  }

  itsMatrixPosition = ++itsMatrixPosition % itsOrder;
}


template<class Type>
void FilterBank<Type>::ReArrangeWithOverlap(blitz::Array<Type, 1> Input)
{
  // The input must be number of bands - number of overlap samples) long
  int PreviousMatrixPosition = (itsMatrixPosition == 0) ? itsOrder - 1 : itsMatrixPosition - 1;

  for (int i = 0; i < itsOverlapSamples; ++i)
  {
    itsReArrangedSignal(i, itsMatrixPosition) = itsReArrangedSignal(itsNumberOfBands - itsOverlapSamples + i,
                                                                    PreviousMatrixPosition);
  }

  for (int i = itsOverlapSamples; i < itsNumberOfBands; ++i)
  {
    itsReArrangedSignal(i, itsMatrixPosition) = Input(i - itsOverlapSamples);
  }
  itsMatrixPosition = ++itsMatrixPosition % itsOrder;
}


template<class Type>
blitz::Array<Type, 1> FilterBank<Type>::Convolve()
{
  blitz::Array<Type, 1> ConvolvedSignal(itsNumberOfBands);
  ConvolvedSignal(blitz::Range(blitz::Range::all())) = 0;

  for (int b = 0; b < itsNumberOfBands; ++b)
  {
    int i = 0;
    for (int o = itsMatrixPosition; o < itsOrder; ++o)
    {
      ConvolvedSignal(b) += itsReArrangedSignal(b, o) * itsFilterCoefficients(b, i++);
    }
    for (int o = 0; o < itsMatrixPosition; ++o)
    {
      ConvolvedSignal(b) += itsReArrangedSignal(b, o) * itsFilterCoefficients(b, i++);
    }
  }
  return ConvolvedSignal;
}


template<>
blitz::Array<complex<float>, 2> FilterBank< complex<float> >::FFT(blitz::Array<complex<float>, 1> ConvolvedSignal)
{
  blitz::Array<complex<float>, 2> FilterBankOutput(itsNumberOfBands, 1);

  fftw_one(fftplancomplex, (fftw_complex*)ConvolvedSignal.data(), (fftw_complex*)FilterBankOutput.data());

  FilterBankOutput /= itsNumberOfBands;

  return FilterBankOutput;
}

template<>
blitz::Array<complex<float>, 2> FilterBank<float>::FFT(blitz::Array<float, 1> ConvolvedSignal)
{
  blitz::Array<complex<float>, 2> FilterBankOutput(itsNumberOfBands, 1);

  rfftwnd_one_real_to_complex(fftplanreal, ConvolvedSignal.data(), (fftw_complex*)FilterBankOutput.data());

  FilterBankOutput /= itsNumberOfBands;

  // mirror the first part of FFT to second part
  FilterBankOutput(blitz::Range(itsNumberOfBands/2+2, itsNumberOfBands - 1), 0)
    = (FilterBankOutput(blitz::Range(1, itsNumberOfBands/2-1), 0)).reverse(0);

  return FilterBankOutput;
}

// template class FilterBank< complex<float> >;
template class FilterBank< float >;




dsp::PolyPhaseFilterbank::PolyPhaseFilterbank ()
{
  filterbank = 0;
  C_filterbank = 0;
  nchan = 0;
}


dsp::PolyPhaseFilterbank::~PolyPhaseFilterbank ()
{
  if (filterbank)
    delete filterbank;

  if (C_filterbank)
    delete C_filterbank;
}


void dsp::PolyPhaseFilterbank::load_coefficients (string filename)
{
  if (filterbank)
    delete filterbank;

  filterbank = new FilterBank<float> (filename);
  nchan = filterbank->getItsNumberOfBands();
}

void dsp::PolyPhaseFilterbank::load_complex_coefficients (string filename)
{
  if (C_filterbank)
    delete C_filterbank;

  C_filterbank = new FilterBank< complex<float> > (filename);
  nchan = C_filterbank->getItsNumberOfBands();
}


//! Perform the convolution transformation on the input TimeSeries
void dsp::PolyPhaseFilterbank::transformation ()
{
  unsigned ndat = input->get_ndat();

  if (input->get_state() == Signal::Nyquist && !filterbank)
    throw Error (InvalidState, "dsp::PolyPhaseFilterbank::transformation",
		 "Real input and no real polyphase filterbank loaded");

  if (input->get_state() == Signal::Analytic && !C_filterbank)
    throw Error (InvalidState, "dsp::PolyPhaseFilterbank::transformation",
		 "Complex input and no complex polyphase filterbank loaded");

  unsigned nsamp_fft = nchan;

  if (verbose)
    cerr << "dsp::PolyPhaseFilterbank::transformation nchan=" << nchan << endl;

  // prepare the output TimeSeries
  output->Observation::operator= (*input);

  // output data will be complex
  output->set_state (Signal::Analytic);

  // output data will be multi-channel
  output->set_nchan (nchan);

  // resize to new number of valid time samples
  output->resize (ndat/nchan);

  double scalefac = 1.0;

  // maybe needs rescaling

  output->rescale (scalefac);
  
  if (verbose) cerr << "dsp::PolyPhaseFilterbank::transformation"
		 " scale=" << output->get_scale() << endl;

  // output data will have new sampling rate
  // NOTE: that nsamp_fft already contains the extra factor of two required
  // when the input TimeSeries is Signal::Nyquist (real) sampled
  double ratechange = 1.0 / double (nsamp_fft);
  output->set_rate (input->get_rate() * ratechange);

  // complex to complex FFT produces a band swapped result
  if (input->get_state() == Signal::Analytic)
    output->set_swap (true);

  // if freq_res is even, then each sub-band will be centred on a frequency
  // that lies on a spectral bin *edge* - not the centre of the spectral bin
  // output->set_dc_centred (freq_res%2);

  // increment the start time by the number of samples dropped from the fft
  // output->change_start_time (nfilt_pos);

  // enable the Response to record its effect on the output Timeseries
  if (response)
    response->mark (output);

  unsigned npol = input->get_npol();

  Signal::State state = input->get_state();

  unsigned out_ndat = output->get_ndat();
  float*   out_ptr = 0;

  // the result of each polyphase filterbank operation
  blitz::Array<complex<float>,2> result;

  for (unsigned ipol=0; ipol<npol; ipol++) {

    float* in_ptr = const_cast<float*>(input->get_datptr (0, ipol));

    unsigned out_iptr = 0;

    // import the data to a Blitz++ array

    for (unsigned idat=0; idat < out_ndat; idat++) {

      if (state == Signal::Nyquist) {
	
	blitz::Array<float,1> A (in_ptr,
				 blitz::shape(nchan*2),
				 blitz::neverDeleteData);
	
	result.reference( filterbank->filter (A) );
	
      } 
      else {
	
	blitz::Array<complex<float>,1> A ((complex<float>*) in_ptr,
					  blitz::shape(nchan),
					  blitz::neverDeleteData);

	result.reference( C_filterbank->filter (A) );
	
      }

      for (unsigned ichan=0; ichan < nchan; ichan++) {

	out_ptr = output->get_datptr (ichan, ipol) + out_iptr;

	blitz::TinyVector<int,2> index (ichan, 0);
	complex<float> z = result(index);

	out_ptr[0] = z.real();
	out_ptr[1] = z.imag();

      }

      // prepare for the next lot
      in_ptr += nchan * 2;
      out_iptr += 2;

    }

  } // for each polarization
    
  if (verbose)
    cerr << "dsp::PolyPhaseFilterbank::transformation exit." << endl;
}

