#include "environ.h"

#include "Error.h"

#include "dsp/TimeSeries.h"

dsp::TimeSeries::TimeSeries()
{
  data = 0;
  size = 0;
  subsize = 0;
  nbit = 32;
}

void dsp::TimeSeries::set_nbit (unsigned)
{
  if (verbose)
    cerr << "dsp::TimeSeries::set_nbit ignored" << endl;
}

//! Allocate the space required to store nsamples time samples.
/*!
  \pre The dimensions of each time sample (nchan, npol, ndim) should
  have been set prior to calling this method.
  \post If: <UL>
  <LI> nsamples == 0, the data buffer is completely de-allocated </LI>
  <LI> nsamples < previous resize(nsamples), the data buffer and its data
  is not modified.  Only the interpretation of the size of each data block
  is changed.
  <LI> nsamples > previous resize(nsamples), the data buffer may be deleted
  and the current data fill be lost.
  </UL>
*/
void dsp::TimeSeries::resize (uint64 nsamples)
{
  if (verbose)
    cerr << "dsp::TimeSeries::resize (" << nsamples << ")" << endl;

  uint64 require = ndim * nsamples * npol * nchan;

  if (!require || require > size) {
    if (data) delete [] data; data = 0;
    size = subsize = 0;
  }

  ndat = nsamples;

  if (!require)
    return;

  if (size == 0) {
    data = new float [require];
    size = require;
  }

  subsize = ndim * nsamples;
}

//! Return pointer to the specified data block
float* dsp::TimeSeries::get_datptr (unsigned ichan, unsigned ipol)
{
  return data + (ichan * npol + ipol) * subsize;
}

//! Return pointer to the specified data block
const float*
dsp::TimeSeries::get_datptr (unsigned ichan, unsigned ipol) const
{
  return data + (ichan * npol + ipol) * subsize;
}


dsp::TimeSeries& dsp::TimeSeries::operator = (const TimeSeries& copy)
{
  if (this == &copy)
    return *this;

  Observation::operator = (copy);
  resize (copy.get_ndat());

  uint64 npt = get_ndat() * get_ndim();

  for (unsigned ichan=0; ichan<get_nchan(); ichan++)
    for (unsigned ipol=0; ipol<get_npol(); ipol++) {

      float* data1 = get_datptr (ichan, ipol);
      const float* data2 = copy.get_datptr (ichan, ipol);

      for (uint64 ipt=0; ipt<npt; ipt++)
        data1[ipt] = data2[ipt];

    }

  return *this;
}

dsp::TimeSeries& dsp::TimeSeries::operator += (const TimeSeries& add)
{
  if (!combinable (add))
    throw Error (InvalidState, "TimeSeries::operator+=",
		 "TimeSeries are not combinable");

  if (get_ndat() != add.get_ndat())
    throw Error (InvalidState, "TimeSeries::operator+=",
		 "ndat="UI64" != "UI64, get_ndat(), add.get_ndat());

  uint64 npt = get_ndat() * get_ndim();

  for (unsigned ichan=0; ichan<get_nchan(); ichan++)
    for (unsigned ipol=0; ipol<get_npol(); ipol++) {

      float* data1 = get_datptr (ichan, ipol);
      const float* data2 = add.get_datptr (ichan, ipol);

      for (uint64 ipt=0; ipt<npt; ipt++)
        data1[ipt] += data2[ipt];

    }

  return *this;
}

void dsp::TimeSeries::zero ()
{
  uint64 npt = get_ndat() * get_ndim();
  for (unsigned ichan=0; ichan<get_nchan(); ichan++)
    for (unsigned ipol=0; ipol<get_npol(); ipol++) {
      float* data = get_datptr (ichan, ipol);
      for (uint64 ipt=0; ipt<npt; ipt++)
        data[ipt]=0.0;
    }
}

//! return value is whether timeseries can still be appended to
bool dsp::TimeSeries::append (const dsp::TimeSeries* little)
{
  bool ret = true;

  uint64 ncontain = get_ndat() * get_ndim();
  uint64 ncopy = little->get_ndat() * little->get_ndim();

  if( verbose ){
    fprintf(stderr,"ncopy="UI64"\n",ncopy);
    fprintf(stderr,"ncontain="UI64"\n",ncontain);
    fprintf(stderr,"subsize="UI64"\n",subsize);
  }

  if( ncontain==subsize )
    throw Error (InvalidRange, "dsp::TimeSeries::append",
		 "Capacity="UI64" is already totally full- none of ndat="UI64" can be appended at all",
		 subsize, ncopy + ncontain);

  if ( subsize <= ncontain + ncopy ){
    ncopy = subsize - ncontain;
    fprintf(stderr,"dsp::TimeSeries::append()- this append will fill up the timeseries from ndat="UI64" with ncopy="UI64" to ndat="UI64".\n",
	    get_ndat(), ncopy, get_ndat()+ncopy/little->get_ndim());
    ret = false;
  }

  if ( get_ndat() == 0 ) {
    Observation::operator=(*little);
    set_ndat (0);
  }    

  else if( !contiguous(*little) )
    throw Error (InvalidState, "dsp::TimeSeries::append",
		 "next TimeSeries is not contiguous");
  

  for( unsigned ichan=0; ichan<nchan; ichan++) {
    for( unsigned ipol=0; ipol<npol; ipol++) {

      const float* from = little->get_datptr (ichan,ipol);
      float* to = get_datptr(ichan,ipol) + ncontain;
      
      memcpy (to, from, ncopy*sizeof(float));
    }
  }

  set_ndat (get_ndat() + ncopy/little->get_ndim());

  return ret;
}

void dsp::TimeSeries::check (float min, float max)
{
  if (get_detected()) {
    min *= fabs(min);
    max *= max;
  }

  min *= scale;
  max *= scale;

  for (unsigned ichan=0; ichan<get_nchan(); ichan++)
    for (unsigned ipol=0; ipol<get_npol(); ipol++) {

      float* data = get_datptr (ichan, ipol);

      for (uint64 idat=0; idat<get_ndat(); idat++)
	for (unsigned idim=0; idim<get_ndim(); idim++) {
	  if (isnan (*data) || *data < min || *data > max)
	    cerr << "dsp::TimeSeries::check data ["
	      "f=" << ichan << ", "
	      "p=" << ipol << ", "
	      "t=" << idat << ", "
	      "d=" << idim << "] = " << *data << endl;
	  data ++;
	}
    }
}
