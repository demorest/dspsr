#include <math.h>
#include <stdlib.h>

#include "window.h"

dsp::window::window() : dsp::shape ()
{
  type = none;
}  

void dsp::window::Hanning (int npts, bool analytic)
{
  Construct (1, npts, analytic);

  float* datptr = buffer;
  float  value = 0.0;

  double denom = npts - 1.0;

  for (int idat=0; idat<ndat; idat++) {
    value = 0.5 * (1 - cos(2.0*M_PI*double(idat)/denom));
    *datptr = value; datptr++;
    if (analytic) {
      *datptr = value; datptr++;
    }
  }
  type = hanning;
}

void dsp::window::set_shape (int npts, Type type, bool analytic)
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

void dsp::window::Welch (int npts, bool analytic)
{
  Construct (1, npts, analytic);

  float* datptr = buffer;
  float  value = 0.0;
  float  tosquare = 0.0;

  float numerator = 0.5 * (npts - 1);
  float denominator = 0.5 * (npts + 1);

  for (int idat=0; idat<ndat; idat++) {
    tosquare = (float(idat)-numerator)/denominator;
    value = 1.0 - tosquare * tosquare;
    *datptr = value; datptr++;
    if (analytic) {
      *datptr = value; datptr++;
    }
  }
  type = welch;
}

void dsp::window::Parzen (int npts, bool analytic)
{
  Construct (1, npts, analytic);

  float* datptr = buffer;
  float  value = 0.0;

  float numerator = 0.5 * (npts - 1);
  float denominator = 0.5 * (npts + 1);

  for (int idat=0; idat<ndat; idat++) {
    value = 1.0 - fabs ((float(idat)-numerator)/denominator);
    *datptr = value; datptr++;
    if (analytic) {
      *datptr = value; datptr++;
    }
  }
  type = parzen;
}


void dsp::window::operate (float* indata, float* outdata) const
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

void dsp::window::normalize()
{
  float* winptr = buffer;

  double total = 0.0;
  for (int idat=0; idat<ndat; idat++) {
    total += *winptr;
    winptr += ndim;
  }

  winptr = buffer;
  int npts = ndat * ndim;
  for (int ipt=0; ipt<npts; ipt++) {
    *winptr /= total;
    winptr ++;
  }
}

double dsp::window::integrated_product (float* data, int incr) const
{
  double total = 0.0;
  int cdat = 0;

  for (int idat=0; idat<ndat; idat++) {
    total += buffer[idat] * data[cdat];
    cdat += incr;
  }

  return total;
}
