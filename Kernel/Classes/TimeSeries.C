#include <memory>

#include "environ.h"
#include "genutil.h"
#include "fsleep.h"

#include "Error.h"

#include "dsp/TimeSeries.h"

dsp::TimeSeries::TimeSeries()
{
  size = 0;
  subsize = 0;
  nbit = 8 * sizeof(float);
}

dsp::TimeSeries::~TimeSeries()
{
  if (data) delete [] data; data = 0;
  size = 0;
  subsize = 0;
}

void dsp::TimeSeries::set_nbit (unsigned _nbit)
{
  if (verbose)
    cerr << "dsp::TimeSeries::set_nbit (" << _nbit << ") ignored" << endl;
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

  if( verbose )
    cerr << "dsp::TimeSeries::resize() has got require="<<require<<endl;

  if (!require || require > size) {
    if (data) delete data; data = 0;
    size = subsize = 0;
  }

  ndat = nsamples;

  if (!require)
    return;

  if (size == 0) {
    data = new float[require];
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

//! return value is number of timesamples actually appended.
//! If it is zero, then none were and we assume 'this' is full
//! If it is nonzero, but not equal to little->get_ndat(), then 'this' is full too
//! If it is equal to little->get_ndat(), it may/may not be full.
uint64 dsp::TimeSeries::append (const dsp::TimeSeries* little)
{
  uint64 ncontain = get_ndat() * get_ndim();
  uint64 ncopy = little->get_ndat() * little->get_ndim();

  if( verbose ){
    fprintf(stderr,"ncopy="UI64"\n",ncopy);
    fprintf(stderr,"ncontain="UI64"\n",ncontain);
    fprintf(stderr,"subsize="UI64"\n",subsize);
  }

  if( ncontain==subsize )
    return 0;

  if ( subsize <= ncontain + ncopy ){
    ncopy = subsize - ncontain;
    fprintf(stderr,"dsp::TimeSeries::append()- this append will fill up the timeseries from ndat="UI64" with ncopy="UI64" to ndat="UI64".\n",
	    get_ndat(), ncopy, get_ndat()+ncopy/little->get_ndim());
  }

  if ( get_ndat() == 0 ) {
    Observation::operator=(*little);
    set_ndat (0);
  }    

  else if( !contiguous(*little) )
    throw Error (InvalidState, "dsp::TimeSeries::append()",
		 "next TimeSeries is not contiguous");
  
  for( unsigned ichan=0; ichan<nchan; ichan++) {
    for( unsigned ipol=0; ipol<npol; ipol++) {

      const float* from = little->get_datptr (ichan,ipol);
      float* to = get_datptr(ichan,ipol) + ncontain;
      
      memcpy (to, from, ncopy*sizeof(float));
    }
  }

  set_ndat (get_ndat() + ncopy/little->get_ndim());

  return ncopy / uint64(ndim);
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

//! Delete the current data buffer and attach to this one
//! This is dangerous as it ASSUMES new data buffer has been pre-allocated and is big enough.  Beware of segmentation faults when using this routine.
//! Also do not try to delete the old memory once you have called this- the TimeSeries::data member now owns it.
void dsp::TimeSeries::attach(auto_ptr<float> _data){
  if( !_data.get() )
    throw Error(InvalidState,"dsp::TimeSeries::attach()",
		"NULL auto_ptr has been passed in- you haven't properly allocated it using 'new' before passing it into this method");

  if (data) delete [] data;
  data = _data.get();
}

bool from_range(unsigned char* fr,const dsp::TimeSeries* tseries){
  unsigned char* fr_min = (unsigned char*)tseries->get_datptr(0,0);
  unsigned char* fr_max = (unsigned char*)(tseries->get_datptr(tseries->get_nchan()-1,tseries->get_npol()-1)+tseries->get_ndat());
  if( fr>=fr_min && fr<fr_max)
    return true;
  fprintf(stderr,"fr_min=%p fr_max=%p fr=%p\n",
	  fr_min,fr_max,fr);
  return false;
}

bool to_range(unsigned char* too,const dsp::TimeSeries* thiz){
  unsigned char* too_min = (unsigned char*)thiz->get_datptr(0,0);
  unsigned char* too_max = (unsigned char*)(thiz->get_datptr(thiz->get_nchan()-1,thiz->get_npol()-1)+thiz->get_ndat());
  if( too>=too_min && too<too_max)
    return true;
  fprintf(stderr,"too_min=%p too_max=%p too=%p\n",
	  too_min,too_max,too);
  return false;
}

//! Copy from the back of 'tseries' into the front of 'this'
void dsp::TimeSeries::copy_from_back(TimeSeries* tseries, uint64 nsamples)
{
  if( !tseries )
    throw Error(InvalidState,"dsp::TimeSeries::copy_from_back()",
		"Null timeseries given as input");

  if( tseries->get_ndat() < nsamples )
    throw Error(InvalidParam,"dsp::TimeSeries::copy_from_back()",
		"Cannot copy "UI64" samples from back of a "UI64" length TimeSeries",
		nsamples,tseries->get_ndat());

  if( !combinable(*tseries) )
    throw Error(InvalidState,"dsp::TimeSeries::copy_from_back()",
		"TimeSeries not combinable");

  fprintf(stderr,"dsp::TimeSeries::copy_from_back(): this: "UI64"/"UI64" nsamples="UI64"\n",
	  ndat,subsize,nsamples);
  fprintf(stderr,"dsp::TimeSeries::copy_from_back(): tseries: "UI64"/"UI64"\n",
	  tseries->ndat,tseries->subsize);

  uint64 bytes_forward = tseries->get_ndat()*ndim*nbit/8;
  uint64 copy_bytes = nsamples*ndim*nbit/8;

  fprintf(stderr,"dsp::TimeSeries::copy_from_back(): bytes_forward="UI64"*%d*%d/8 = "UI64" copy_bytes="UI64"\n",
	  tseries->get_ndat(),ndim,nbit,bytes_forward,copy_bytes);

  for( unsigned ichan=0; ichan<nchan; ichan++){
    for( unsigned ipol=0; ipol<npol; ipol++){
      unsigned char* from = ((unsigned char*)tseries->get_datptr(ichan,ipol)) + bytes_forward - copy_bytes;

      unsigned char* to = (unsigned char*)get_datptr(ichan,ipol);
      //fprintf(stderr,"dsp::TimeSeries::copy_from_back(): ichan=%d ipol=%d ndat="UI64" to=%p\n",
      //      ichan,ipol,ndat,to);

      if( !from_range(from+copy_bytes-1,tseries) || !to_range(to+copy_bytes-1,this) ){
	fprintf(stderr,"Error: ichan=%d ipol=%d from=%p to=%p\n",
		ichan,ipol,from,to);
	exit(-1);
      }

      memcpy(to,from,copy_bytes);
    }
  }
}

//! Copy from the front of 'tseries' into the front of 'this'
void dsp::TimeSeries::copy_from_front(TimeSeries* tseries, uint64 nsamples)
{
  if( !tseries )
    throw Error(InvalidState,"dsp::TimeSeries::copy_from_front()",
		"Null timeseries given as input");

  if( tseries->get_ndat() < nsamples )
    throw Error(InvalidParam,"dsp::TimeSeries::copy_from_front()",
		"Cannot copy "UI64" samples from front of a "UI64" length TimeSeries",
		nsamples,tseries->get_ndat());

  if( !combinable(*tseries) )
    throw Error(InvalidState,"dsp::TimeSeries::copy_from_back()",
		"TimeSeries not combinable");

  uint64 copy_bytes = nsamples*ndim*nbit/8;

  for( unsigned ichan=0; ichan<nchan; ichan++){
    for( unsigned ipol=0; ipol<npol; ipol++){
      const unsigned char* from = (const unsigned char*)tseries->get_datptr(ichan,ipol);
      unsigned char* to = (unsigned char*)get_datptr(ichan,ipol);

      memcpy(to,from,copy_bytes);
    }
  }
}

void dsp::TimeSeries::rotate_backwards_onto(TimeSeries* ts){
  if( !ts )
    throw Error(InvalidState,"dsp::TimeSeries::rotate_backwards_onto()",
		"Null timeseries given as input");
  
  if( ts->get_ndat() > ndat )
    throw Error(InvalidState,"dsp::TimeSeries::rotate_backwards_onto()",
		"You must supply a TimeSeries that has fewer timesamples in it than 'this'");

  if( !ts->contiguous(*this) )
    throw Error (InvalidState, "dsp::TimeSeries::rotate_backwards_onto()",
		 "TimeSeries'es are not contiguous");   

  vector<float> storage( (ndat - ts->ndat)*ndim );
  fprintf(stderr,"dsp::TimeSeries::rotate_backwards_onto() ndat="UI64" ts->ndat="UI64" ndim=%d so storage.size()=%d\n",
	  ndat,ts->ndat,ndim,storage.size());

  for( unsigned ichan=0; ichan<nchan; ichan++){
    for( unsigned ipol=0; ipol<npol; ipol++){
      
      // Step (1) rotate samples forward.  (The lazy memory inefficient way)
      const float* from1 = get_datptr(ichan,ipol);
      float* to1 = storage.begin();
      memcpy(to1,from1,storage.size()*sizeof(float));

      const float* from2 = storage.begin();
      float* to2 = get_datptr(ichan,ipol) + ts->ndat*ndim;
      memcpy(to2,from2,storage.size()*sizeof(float));

      // Step (2) copy in from ts
      const float* from3 = ts->get_datptr(ichan,ipol);
      float* to3 = get_datptr(ichan,ipol);
      memcpy(to3,from3,ts->ndat*ts->ndim*sizeof(float));
    }
  }
  
  // Step (3) Clean up
  uint64 _ndat = ndat;
  Observation::operator=( *ts );
  set_ndat( _ndat );

  return;
}
