//  PolyPhaseFilterbank.cc:
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
PolyPhaseFilterbank<Type>::PolyPhaseFilterbank(string CoefficientFile, int OverlapSamples, int Real)
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
PolyPhaseFilterbank<Type>::~PolyPhaseFilterbank()
{
  fftw_destroy_plan(fftplancomplex);
  rfftwnd_destroy_plan(fftplanreal);
}


template<class Type>
Array<complex<float>, 2> PolyPhaseFilterbank<Type>::filter(Array<Type, 1> Input)
{
  Array<complex<float>, 2> PolyPhaseFilterbankOutput(itsNumberOfBands, 1);
  Array<complex<float>, 1> Temp(itsNumberOfBands / 2);

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
  Array<Type, 1> ConvolvedSignal = Convolve();

  // Do a FFT n the convolved signal and return the output
  PolyPhaseFilterbankOutput = FFT(ConvolvedSignal);

  return PolyPhaseFilterbankOutput;
}


template<class Type>
void PolyPhaseFilterbank<Type>::ReArrange(Array<Type, 1> Input) // This function might be not necessary instead use ReArrange with overlap with overlap = 0
{
  // The input must be number of bands long!
  for (int i = 0; i < itsNumberOfBands; ++i)
  {
    itsReArrangedSignal(i, itsMatrixPosition) = Input(i);
  }

  itsMatrixPosition = ++itsMatrixPosition % itsOrder;
}


template<class Type>
void PolyPhaseFilterbank<Type>::ReArrangeWithOverlap(Array<Type, 1> Input)
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
Array<Type, 1> PolyPhaseFilterbank<Type>::Convolve()
{
  Array<Type, 1> ConvolvedSignal(itsNumberOfBands);
  ConvolvedSignal(Range(Range::all())) = 0;

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
Array<complex<float>, 2> PolyPhaseFilterbank< complex<float> >::FFT(Array<complex<float>, 1> ConvolvedSignal)
{
  Array<complex<float>, 2> PolyPhaseFilterbankOutput(itsNumberOfBands, 1);

  fftw_one(fftplancomplex, (fftw_complex*)ConvolvedSignal.data(), (fftw_complex*)PolyPhaseFilterbankOutput.data());

  PolyPhaseFilterbankOutput /= itsNumberOfBands;

  return PolyPhaseFilterbankOutput;
}

template<>
Array<complex<float>, 2> PolyPhaseFilterbank<float>::FFT(Array<float, 1> ConvolvedSignal)
{
  Array<complex<float>, 2> PolyPhaseFilterbankOutput(itsNumberOfBands, 1);

  rfftwnd_one_real_to_complex(fftplanreal, ConvolvedSignal.data(), (fftw_complex*)PolyPhaseFilterbankOutput.data());

  PolyPhaseFilterbankOutput /= itsNumberOfBands;

  // mirror the first part of FFT to second part
  PolyPhaseFilterbankOutput(Range(itsNumberOfBands / 2 + 2, itsNumberOfBands - 1), 0) =
              (PolyPhaseFilterbankOutput(Range(1, itsNumberOfBands / 2 - 1), 0)).reverse(0);

  return PolyPhaseFilterbankOutput;
}

// template class PolyPhaseFilterbank< complex<float> >;
template class PolyPhaseFilterbank< float >;
