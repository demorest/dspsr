#include <assert.h>

#include "dsp/Shape.h"
#include "genutil.h"
#include "Error.h"

bool dsp::Shape::verbose = false;

void dsp::Shape::init ()
{
  buffer = NULL;
  bufsize = 0;
  offset = 0;

  npol = nchan = ndat = ndim = 0;
  borrowed = false;
}

void dsp::Shape::destroy ()
{
  if (!borrowed && buffer!=NULL) delete [] buffer; buffer = NULL;
  init ();
}

dsp::Shape::Shape(void) {
  init ();
}

dsp::Shape::~Shape(void) {
  destroy ();
}

//! Copy constructor
dsp::Shape::Shape (const Shape& copy)
{
  init ();
  Shape::operator = (copy);
}

//! Assignment operator
const dsp::Shape& dsp::Shape::operator = (const Shape& copy)
{
  if (&copy == this)
    return *this;

  resize (copy.npol, copy.nchan, copy.ndat, copy.ndim);

  unsigned pts = ndim * ndat * nchan;

  for (unsigned ipol=0; ipol < npol; ipol++) {
    float* buf = buffer + offset * ipol;
    float* dbuf = copy.buffer + copy.offset * ipol;
    for (unsigned ipt=0; ipt<pts; ipt++) {
      *buf = *dbuf;
      buf++; dbuf++;
    }
  }

  return *this;
}

const dsp::Shape& dsp::Shape::operator /= (float factor)
{
  return operator *= (1.0/factor);
}

const dsp::Shape& dsp::Shape::operator *= (float factor)
{
  unsigned pts = ndim * ndat * nchan;

  for (unsigned ipol=0; ipol < npol; ipol++) {
    float* buf = buffer + offset * ipol;
    for (unsigned ipt=0; ipt<pts; ipt++)
      buf[ipt] *= factor;
  }
  return *this;
}

const dsp::Shape& dsp::Shape::operator += (const dsp::Shape& ds)
{
  if (npol != ds.npol)
    throw_str ("dsp::Shape::operator += npol=%d!=ds.npol=%d", npol,ds.npol);
  if (nchan != ds.nchan)
    throw_str ("dsp::Shape::operator += nchan=%d!=ds.nchan=%d",nchan,ds.nchan);
  if (ndat != ds.ndat)
    throw_str ("dsp::Shape::operator += ndat=%d!=ds.ndat=%d", ndat,ds.ndat);
  if (ndim != ds.ndim)
    throw_str ("dsp::Shape::operator += ndim=%d!=ds.ndim=%d", ndim,ds.ndim);

  unsigned pts = ndim * ndat * nchan;

  for (unsigned ipol=0; ipol < npol; ipol++) {
    float* buf = buffer + offset * ipol;
    float* dbuf = ds.buffer + ds.offset * ipol;
    for (unsigned ipt=0; ipt<pts; ipt++) {
      *buf += *dbuf;
      buf++; dbuf++;
    }
  }
  return *this;
}

void dsp::Shape::resize (unsigned _npol, unsigned _nchan,
			 unsigned _ndat, unsigned _ndim)
{
  if (verbose)
    cerr <<
      "dsp::Shape::resize"
      "  npol=" << _npol <<
      "  nchan=" << _nchan <<
      "  ndat=" << _ndat <<
      "  ndim=" << _ndim << endl;

  npol = _npol;
  nchan = _nchan;
  ndat = _ndat;
  ndim = _ndim;

  offset = ndim * ndat * nchan;

  unsigned new_bufsize = ndim * ndat * nchan * npol;

  // only reallocate the array if necessary
  // BAD BAD BAD- dedispersion assumes contiguity in
  // MultiFilterbank::transformation() and Dedispersion::operate()
  //  if (!borrowed && bufsize >= new_bufsize)
  //return;

  if (!borrowed && buffer!=NULL) delete [] buffer; buffer = NULL;

  borrowed = false;

  buffer = new float [new_bufsize]; assert (buffer != NULL);
  bufsize = new_bufsize;
}


bool dsp::Shape::matches (const Shape* shape)
{
  return
    npol == shape->get_npol() &&
    nchan == shape->get_nchan() &&
    ndat == shape->get_ndat();
}

void dsp::Shape::borrow (const dsp::Shape& data, unsigned ichan)
{
  if (ichan >= data.nchan)
    throw_str ("dsp::Shape::external invalid ichan=%d/%d", ichan, data.nchan);

  // cerr << "dsp::Shape::external ichan=" << ichan << "/" << nchan << endl;
  destroy ();

  ndim = data.ndim;
  ndat = data.ndat;
  nchan = 1;
  npol = data.npol;

  offset = data.offset;
  buffer = data.buffer + ichan * ndat * ndim;

  borrowed = true;
}

void dsp::Shape::scrunch_to (unsigned new_ndat)
{
  if (ndat == new_ndat)
    return;

  if (borrowed)
    throw_str ("dsp::Shape::scrunch_to cannot scrunch borrowed data");

  if (new_ndat == 0)
    throw_str ("dsp::Shape::scrunch_to invalid ndat=0");

  if (ndat < new_ndat)
    throw_str ("dsp::Shape::scrunch_to ndat=%d > current ndat=%d",
	       new_ndat, ndat);

  if (ndat % new_ndat)
    throw_str ("dsp::Shape::scrunch_to uneven factor");

  unsigned sfactor = ndat / new_ndat;

  unsigned npts = ndim * new_ndat * nchan;

  register float* mvr = buffer;
  register float* adr = buffer;

  for (unsigned ipol=0; ipol < npol; ipol++) {
    for (unsigned ipt=0; ipt<npts; ipt++) {
      *mvr = *adr; adr++;
      for (unsigned ipta=1; ipta<sfactor; ipta++) {
	*mvr += *adr; adr++;
      }
      mvr ++;
    }
  }

  ndat = new_ndat;
  offset = npts;
}


//! Rotate data so that Shape[i] = Shape[i+npt]
// restriction: ndat % npt must equal zero
void dsp::Shape::rotate (int rotbin)
{
  if (verbose)
    cerr << "dsp::Shape::rotate ("<< rotbin <<")" << endl;

  unsigned pts = nchan * ndat * ndim;
  unsigned nrot = abs(rotbin) * ndim;
  unsigned nstep = pts / nrot - 1;

  if (nrot > pts)
    throw Error (InvalidParam, "dsp::Shape::rotate",
		 "abs(rotbin=%d) > nchan=%d*ndat=%d", rotbin, nchan, ndat);

  rotbin *= ndim;

  float temp = 0;

  float* p1=0;
  float* p2=0;

  for (unsigned ipol=0; ipol < npol; ipol++) {

    float* buf = buffer + offset * ipol;

    for (unsigned irot=0; irot<nrot; irot++) {

      temp = buf[irot];
      int ipt = 0;
  
      for (unsigned istep=0; istep<nstep; istep++) {
	
	p1 = buf + (irot + ipt + pts) % pts;
	p2 = buf + (irot + ipt + rotbin + pts) % pts;

	*p1 = *p2;

        ipt += rotbin;

      }

      *p2 = temp;

    }
  }
}

void dsp::Shape::zero ()
{
  for (unsigned ifilt=0; ifilt<bufsize; ifilt++)
    buffer[ifilt] = 0;
}
