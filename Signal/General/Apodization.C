#include <math.h>
#include <stdlib.h>

#include "Apodization.h"

dsp::Apodization::Apodization()
{
  type = none;
}  

void dsp::Apodization::Hanning (int npts, bool analytic)
{
  resize (1, 1, npts, (analytic)?2:1);

  float* datptr = buffer;
  float  value = 0.0;

  double denom = npts - 1.0;

  for (unsigned idat=0; idat<ndat; idat++) {
    value = 0.5 * (1 - cos(2.0*M_PI*double(idat)/denom));
    *datptr = value; datptr++;
    if (analytic) {
      *datptr = value; datptr++;
    }
  }
  type = hanning;
}

void dsp::Apodization::set_shape (int npts, Type type, bool analytic)
{
  switch (type)  {
  case hanning:
    Hanning (npts, analytic);
    break;
  case welch:
    Welch (npts, analytic);
    break;
  case parzen:
    Parzen (npts, analytic);
    break;
  case none:
  default:
    break;
  }
}

void dsp::Apodization::Welch (int npts, bool analytic)
{
  resize (1, 1, npts, (analytic)?2:1);

  float* datptr = buffer;
  float  value = 0.0;
  float  tosquare = 0.0;

  float numerator = 0.5 * (npts - 1);
  float denominator = 0.5 * (npts + 1);

  for (unsigned idat=0; idat<ndat; idat++) {
    tosquare = (float(idat)-numerator)/denominator;
    value = 1.0 - tosquare * tosquare;
    *datptr = value; datptr++;
    if (analytic) {
      *datptr = value; datptr++;
    }
  }
  type = welch;
}

void dsp::Apodization::Parzen (int npts, bool analytic)
{
  resize (1, 1, npts, (analytic)?2:1);

  float* datptr = buffer;
  float  value = 0.0;

  float numerator = 0.5 * (npts - 1);
  float denominator = 0.5 * (npts + 1);

  for (unsigned idat=0; idat<ndat; idat++) {
    value = 1.0 - fabs ((float(idat)-numerator)/denominator);
    *datptr = value; datptr++;
    if (analytic) {
      *datptr = value; datptr++;
    }
  }
  type = parzen;
}


void dsp::Apodization::operate (float* indata, float* outdata) const
{
  int npts = ndat * ndim;
  float* winptr = buffer;

  if (outdata == NULL)
    outdata = indata;
  
  for (int ipt=0; ipt<npts; ipt++) {
    *outdata = *indata * *winptr;
    outdata ++; indata++; winptr++;
  } 
}

void dsp::Apodization::normalize()
{
  float* winptr = buffer;

  double total = 0.0;
  for (unsigned idat=0; idat<ndat; idat++) {
    total += *winptr;
    winptr += ndim;
  }

  winptr = buffer;
  unsigned npts = ndat * ndim;
  for (unsigned ipt=0; ipt<npts; ipt++) {
    *winptr /= total;
    winptr ++;
  }
}

double dsp::Apodization::integrated_product (float* data, unsigned incr) const
{
  double total = 0.0;
  unsigned cdat = 0;

  for (unsigned idat=0; idat<ndat; idat++) {
    total += buffer[idat] * data[cdat];
    cdat += incr;
  }

  return total;
}
