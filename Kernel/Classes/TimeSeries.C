#include <stdio.h>

#include <algorithm>
#include <memory>

#include "environ.h"
#include "genutil.h"
#include "fsleep.h"
#include "minmax.h"

#include "Error.h"

#include "dsp/TimeSeries.h"

bool operator==(const dsp::ChannelPtr& c1, const dsp::ChannelPtr& c2)
{ return c1.ts==c2.ts && c1.ptr==c2.ptr && c1.ichan==c2.ichan && c1.ipol==c2.ipol; } 
bool operator!=(const dsp::ChannelPtr& c1, const dsp::ChannelPtr& c2)
{ return !(c1==c2); }

bool dsp::TimeSeriesPtr::operator < (const TimeSeriesPtr& tsp) const{
  if( !ptr || !tsp.ptr )
    throw Error(InvalidParam,"dsp::TimeSeriesPtr::operator<(TimeSeriesPtr&))"
		"Null pointer passed in");
  return ptr->get_centre_frequency() < tsp.ptr->get_centre_frequency();
}

bool dsp::ChannelPtr::operator < (const ChannelPtr& c) const{
  return ts->get_centre_frequency(ichan) < c.ts->get_centre_frequency(c.ichan);
}

dsp::TimeSeries::TimeSeries()
{
  init();
}

void dsp::TimeSeries::init(){
  Observation::init();
  
  data = buffer = NULL;
  size = 0;
  subsize = 0;
  nbit = 8 * sizeof(float);
}
  

dsp::TimeSeries::TimeSeries(const TimeSeries& ts) {
  init();
  operator=(ts);
}

dsp::TimeSeries* dsp::TimeSeries::clone(){
  return new TimeSeries(*this);
}

dsp::TimeSeries::~TimeSeries()
{
  if (buffer) delete [] buffer; buffer = 0;
  data = 0;
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

  uint64 require = uint64(ndim) * nsamples * uint64(npol) * uint64(nchan);

  if( verbose )
    fprintf(stderr,"dsp::TimeSeries::resize require= "UI64" * "UI64" * "UI64" * "UI64" * = "UI64" (cf size="UI64") (buffer=%p and data=%p\n",
	    uint64(ndim) , nsamples , uint64(npol) , uint64(nchan), require, size,
	    buffer,data);

  if (!require || require > size) {
    if (buffer) delete buffer; buffer = 0;
    data = 0;
    size = subsize = 0;

    if( verbose )
      fprintf(stderr,"dsp::TimeSeries::resize has deleted old data buffer, size now is zero\n");
  }

  ndat = nsamples;

  if (!require)
    return;

  if (size == 0) {
    if( verbose ) fprintf(stderr,"dsp::TimeSeries::resize() calling new float["UI64"]\n", require);
    buffer = new float[require];
    if( verbose ) fprintf(stderr,"dsp::TimeSeries::resize() have called new float["UI64"] buffer=%p\n", require, buffer);
    size = require;
  }

  subsize = ndim * nsamples;

  data = buffer;

  if( verbose )
    fprintf(stderr,"dsp::TimeSeries::resize() returning with data=%p\n",data);
}

//! Offset the base pointer by offset time samples
void dsp::TimeSeries::seek (int64 offset)
{
  if (offset == 0)
    return;

  if (offset > int64(ndat))
    throw Error (InvalidRange, "dsp::TimeSeries::seek",
		 "offset=%d > ndat=%d", offset, ndat);

  int64 current_offset = int64(data - buffer) * int64(ndim);

  if (-offset > current_offset)
    throw Error (InvalidRange, "dsp::TimeSeries::seek",
		 "offset=%d > current_offset=%d", offset, current_offset);

  data += offset * int64(ndim);
  ndat -= offset;
  change_start_time (offset);
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

double dsp::TimeSeries::mean (unsigned ichan, unsigned ipol)
{
  if (get_ndim() != 1)
    throw Error (InvalidState, "dsp::TimeSeries::mean", "ndim != 1");

  float* data = get_datptr (ichan, ipol);
  double mean = 0.0;
  
  for (unsigned idat=0; idat<ndat; idat++)
    mean += data[idat];

  return mean/double(ndat);
}

//! Return a pointer to the ichan'th frequency ordered channel and pol
float* dsp::TimeSeries::get_ordered_datptr(unsigned ichan,unsigned ipol){
  if( !swap )
    return get_datptr(ichan,ipol);
  
  if(ichan<nchan/2)
    return get_datptr(ichan+nchan/2,ipol);
  return get_datptr(ichan-nchan/2,ipol);
}

dsp::TimeSeries& dsp::TimeSeries::operator = (const TimeSeries& copy)
{
  if (this == &copy)
    return *this;

  Observation::operator = (copy);
  nbit = copy.get_nbit();

  resize (copy.get_ndat());

  uint64 npt = get_ndat() * get_ndim();

  for (unsigned ichan=0; ichan<get_nchan(); ichan++){
    for (unsigned ipol=0; ipol<get_npol(); ipol++) {
      float* data1 = get_datptr (ichan, ipol);
      const float* data2 = copy.get_datptr (ichan, ipol);

      for (uint64 ipt=0; ipt<npt; ipt++)
        data1[ipt] = data2[ipt];
    }
  }

  return *this;
}

dsp::TimeSeries& dsp::TimeSeries::operator += (const TimeSeries& add)
{
  if( get_ndat()==0 )
    return operator=( add );

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

dsp::TimeSeries& dsp::TimeSeries::operator *= (float mult){
  if( verbose )
    fprintf(stderr,"In dsp::TimeSeries::operator*=()\n");
  //  exit(0);

  if( fabs(mult-1.0) < 1.0e-9 )
    return *this;

  for (unsigned ichan=0; ichan<get_nchan(); ichan++){
    for (unsigned ipol=0; ipol<get_npol(); ipol++) {
      float* dat = get_datptr(ichan,ipol);
      unsigned npts = unsigned(get_ndat()*get_ndim());

      for( unsigned i=0; i<npts; ++i)
	dat[i] *= mult;
    }
  }
  
  rescale( mult );

  return *this;
}

void dsp::TimeSeries::zero ()
{
  uint64 npt = get_ndat() * get_ndim();
  for (unsigned ichan=0; ichan<get_nchan(); ichan++)
    for (unsigned ipol=0; ipol<get_npol(); ipol++) {
      float* dat = get_datptr (ichan, ipol);
      for (uint64 ipt=0; ipt<npt; ipt++)
        dat[ipt]=0.0;
    }
}

//! Convenience function for initialising new TimeSeries from 2 old ones
void dsp::TimeSeries::hack_together(TimeSeries* band1, TimeSeries* band2){
  vector<TimeSeries*> buddies(2);
  buddies[0] = band1;
  buddies[1] = band2;

  hack_together(buddies);
}

//! Hack together 2 different bands (not pretty)
void dsp::TimeSeries::hack_together(vector<TimeSeries*> bands){
  if( bands.size()==1 ){
    operator=( *bands[0] );
    return;
  }

  //verbose = true;
  if( verbose ){
    fprintf(stderr,"In dsp::TimeSeries::hack_together() with bands at %f and %f MHz\n",
	    bands[0]->get_centre_frequency(),bands[1]->get_centre_frequency());
    fprintf(stderr,"The bands are %p and %p\n",bands[0],bands[1]);
  }
  if( bands.empty() )
    throw Error(InvalidState,"dsp::TimeSeries::hack_together()",
		"input vector is empty");
  for( unsigned i=0; i<bands.size(); i++){
    if( !bands[i] )
      throw Error(InvalidParam,"dsp::TimeSeries::hack_together()",
		  "Null TimeSeries passed into this method! (%d)",i);
    if( bands[i]==this )
      throw Error(InvalidParam,"dsp::TimeSeries::hack_together()",
		  "You are trying to do an inplace hack_together.  The input bands must come from different TimeSeries instantiations to the output TimeSeries");
  }

  // Sort the TimeSeries's
  vector<TimeSeriesPtr> to_sort(bands.size());
  for( unsigned i=0; i<bands.size(); i++)
    to_sort[i].ptr = bands[i];

  sort(to_sort.begin(),to_sort.end());

  if( bands[0]->get_bandwidth() < 0.0 ){
    vector<TimeSeriesPtr> temp(bands.size());
    unsigned j=temp.size()-1;
    for( unsigned i=0; i<to_sort.size(); i++,j--)
      temp[j] = to_sort[i];
    for( unsigned i=0; i<to_sort.size(); i++)
      to_sort[i] = temp[i];
  }

  for( unsigned i=0; i<bands.size(); i++)
    bands[i] = to_sort[i].ptr;

  // Check the sorted TimeSeries's are combinable
  for( unsigned iband=1; iband<bands.size(); iband++){
    if( !bands[iband-1]->combinable(*bands[iband],true) )
      throw Error(InvalidParam,"dsp::TimeSeries::hack_together()",
		  "bands %d and %d aren't combinable",
		  iband-1,iband);
  } 

  if( bands.size()==1 ){
    operator=( *bands.front() );
    return;
  }
  
  Observation::operator=( *bands.front() );
  set_nchan( get_nchan()*bands.size() );
  resize( get_ndat() );

  unsigned chans_per_iband = bands.front()->get_nchan();

  const float* from;
  float* to;

  for( unsigned iband=0; iband<bands.size(); iband++){
    for( unsigned ichan=0; ichan<bands[iband]->get_nchan(); ichan++){
      for( unsigned ipol=0; ipol<get_npol(); ipol++){
	if( !swap ){
	  from = bands[iband]->get_datptr(ichan,ipol);
	  to = get_datptr(iband*chans_per_iband+ichan,ipol);
	}
	else{
	  int offset_chans = chans_per_iband/2;
	  if( ichan > chans_per_iband/2 )
	    offset_chans = -chans_per_iband/2;

	  from = bands[iband]->get_datptr(ichan+offset_chans,ipol);
	  to = get_datptr(iband*chans_per_iband+ichan,ipol);
	}
	memcpy(to, from, get_ndat()*get_ndim()*get_nbit()/8);
      }
    }
  }

  set_bandwidth( get_bandwidth()*bands.size() );
  
  vector<double> cfreqs(bands.size());
  for( unsigned iband=0; iband<bands.size(); iband++){
    cfreqs[iband] = bands[iband]->get_centre_frequency();
    //    fprintf(stderr,"bands[%d]->cf=%f\n",
    //    iband,bands[iband]->get_centre_frequency());
  }

  double min_cfreq = findmin(&(cfreqs[0]),&(cfreqs[0])+cfreqs.size());
  double max_cfreq = findmax(&(cfreqs[0]),&(cfreqs[0])+cfreqs.size());

  set_centre_frequency(0.5*(max_cfreq-min_cfreq)+min_cfreq);
  //fprintf(stderr,"cf=0.5*(%f-%f)+%f = %f\n",
  //  max_cfreq,min_cfreq,min_cfreq,get_centre_frequency());
  //exit(0);
  set_swap( false );

  if( verbose )
    fprintf(stderr,"Exiting from dsp::TimeSeries::hack_together()\n");
}

//! return value is number of timesamples actually appended.
//! If it is zero, then none were and we assume 'this' is full
//! If it is nonzero, but not equal to little->get_ndat(), then 'this' is full too
//! If it is equal to little->get_ndat(), it may/may not be full.
uint64 dsp::TimeSeries::append (const dsp::TimeSeries* little)
{
  uint64 ncontain = get_ndat() * get_ndim();
  uint64 ncopy = little->get_ndat() * little->get_ndim();

  append_checks(ncontain,ncopy,little);
  if( ncontain==subsize )
    return 0;

  if ( get_ndat() == 0 ) {
    Observation::operator=(*little);
    set_ndat (0);
  }    
  else if( !contiguous(*little,true) )
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

uint64 dsp::TimeSeries::append(const dsp::TimeSeries* little,unsigned ichan,unsigned ipol){
  uint64 ncontain = get_ndat() * get_ndim();
  uint64 ncopy = little->get_ndat() * little->get_ndim();
  
  append_checks(ncontain,ncopy,little);
  if( ncontain==subsize )
    return 0;

  if ( get_ndat() == 0 ) {
    Observation::operator=(*little);
    set_nchan(1);
    set_npol(1);
    set_centre_frequency( little->get_centre_frequency( ichan ) );
    set_bandwidth( little->get_bandwidth()/little->get_nchan() );
    set_dc_centred( true );
    set_ndat (0);
  }    
  else if( !contiguous(*little,true,ichan,ipol) )
    throw Error (InvalidState, "dsp::TimeSeries::append()",
		 "next TimeSeries is not contiguous");
  
  const float* from = little->get_datptr (ichan,ipol);
  float* to = get_datptr(0,0) + ncontain;
  memcpy (to, from, ncopy*sizeof(float));
  
  set_ndat (get_ndat() + ncopy/little->get_ndim());

  return ncopy / uint64(ndim);
}			       

void dsp::TimeSeries::append_checks(uint64& ncontain,uint64& ncopy,
				    const TimeSeries* little){
  if( verbose ){
    fprintf(stderr,"ncopy="UI64"\n",ncopy);
    fprintf(stderr,"ncontain="UI64"\n",ncontain);
    fprintf(stderr,"subsize="UI64"\n",subsize);
  }

  if ( subsize <= ncontain + ncopy ){
    ncopy = subsize - ncontain;
    fprintf(stderr,"dsp::TimeSeries::append()- this append will fill up the timeseries from ndat="UI64" with ncopy="UI64" to ndat="UI64".\n",
	    get_ndat(), ncopy, get_ndat()+ncopy/little->get_ndim());
  }
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

      float* dat = get_datptr (ichan, ipol);

      for (uint64 idat=0; idat<get_ndat(); idat++)
	for (unsigned idim=0; idim<get_ndim(); idim++) {
	  if (isnan (*dat) || *dat < min || *dat > max)
	    cerr << "dsp::TimeSeries::check data ["
	      "f=" << ichan << ", "
	      "p=" << ipol << ", "
	      "t=" << idat << ", "
	      "d=" << idim << "] = " << *dat << endl;
	  dat ++;
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

  if (buffer) delete [] buffer; buffer = 0;
  buffer = data = _data.release();
}

//! Call this when you do not want to transfer ownership of the array
void dsp::TimeSeries::attach(float* _data){
  if( !_data )
    throw Error(InvalidState,"dsp::TimeSeries::attach()",
		"NULL ptr has been passed in- you haven't properly allocated it using 'new' before passing it into this method");

  if (buffer) delete [] buffer; buffer = 0;
  buffer = data = _data;
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

//! Calculates the mean and the std dev of the timeseries, removes the mean, and scales to sigma
void dsp::TimeSeries::normalise(){
  if( ndim!=1 )
    throw Error(InvalidState,"dsp::TimeSeries::normalise()",
		"You can only give a real timeseries to this method- ie ndim!=1");

  for( unsigned ichan=0; ichan<nchan; ichan++){
    for( unsigned ipol=0; ipol<npol; ipol++){
      float* dat = get_datptr(ichan,ipol);
      float mmean = many_findmean(dat,dat+ndat);

      float ssigma = sqrt( many_findvar( dat, dat+ndat, mmean) );

      register unsigned the_ndat = ndat;

      // HSK TODO: which is quicker- 2 loops or 1???  
      for( unsigned i=0; i<the_ndat; ++i)
	dat[i] -= mmean;

      for( unsigned i=0; i<the_ndat; ++i)
	dat[i] /= ssigma;

    }
  }

}

//! Returns the maximum bin of channel 0, pol 0
unsigned dsp::TimeSeries::get_maxbin(){
  return find_binmax(get_datptr(0,0),get_datptr(0,0)+get_ndat());
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
      float* to1 = &(storage[0]);
      memcpy(to1,from1,storage.size()*sizeof(float));

      const float* from2 = &(storage[0]);
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
