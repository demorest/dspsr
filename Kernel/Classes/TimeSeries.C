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

dsp::TimeSeries::TimeSeries() : DataSeries() {
  init();
}
  
dsp::TimeSeries::TimeSeries(const TimeSeries& ts) : DataSeries() {
  operator=(ts);
}

void dsp::TimeSeries::init(){
  DataSeries::init();
  nbit = 8 * sizeof(float);
  data = 0;
}

dsp::TimeSeries* dsp::TimeSeries::clone(){
  return new TimeSeries(*this);
}

dsp::TimeSeries* dsp::TimeSeries::null_clone(){
  return new TimeSeries;
}

dsp::TimeSeries::~TimeSeries(){ }

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
  DataSeries::resize(nsamples);

  data = (float*)buffer;
}

//! Use the supplied array to store nsamples time samples.
//! Always deletes any existing data
void dsp::TimeSeries::resize( uint64 nsamples, uint64 bytes_supplied, unsigned char* buffer_supplied){
  uint64 require = uint64(get_ndim()) * nsamples * uint64(npol) * uint64(nchan);

  if( require > bytes_supplied )
    throw Error(InvalidState,"dsp::TimeSeries::resize()",
		"Not enough bytes supplied ("UI64") to fit in "UI64" samples -> "UI64" bytes",
		bytes_supplied, nsamples, require );

  resize(0);

  buffer = buffer_supplied;
  data = (float*)buffer_supplied;
  size = bytes_supplied;
  set_ndat( nsamples );
  subsize = (get_ndim() * nsamples * get_nbit())/8;
}

//! Equivalent to resize(0) but instead of deleting data, returns the pointer for reuse elsewhere
void dsp::TimeSeries::zero_resize(unsigned char*& _buffer, uint64& nbytes){
  _buffer = (unsigned char*)buffer;
  nbytes = size;

  buffer = 0;
  data = 0;
  resize(0);
}

//! Offset the base pointer by offset time samples
void dsp::TimeSeries::seek (int64 offset)
{
  if (offset == 0)
    return;

  if (offset > int64(get_ndat()))
    throw Error (InvalidRange, "dsp::TimeSeries::seek",
		 "offset=%d > ndat=%d In English: you've tried to seek past the end of the data", offset, get_ndat());

  float* fbuffer = (float*)buffer;

  int64 current_offset = int64(data - fbuffer) * int64(get_ndim());

  if (-offset > current_offset)
    throw Error (InvalidRange, "dsp::TimeSeries::seek",
		 "offset=%d > current_offset=%d In English: you've tried to seek before the start of the data", offset, current_offset);

  data += offset * int64(get_ndim());
  set_ndat( get_ndat() - offset );
  change_start_time (offset);
}

//! Returns a uchar pointer to the first piece of data
unsigned char* dsp::TimeSeries::get_data(){
  if( !data && !buffer )
    throw Error(InvalidState,"dsp::TimeSeries::get_data()",
		"Neither data nor buffer is defined");
  if( !data )
    data = (float*)buffer;
  return ((unsigned char*)data);
}

//! Returns a uchar pointer to the first piece of data
const unsigned char* dsp::TimeSeries::const_get_data() const{
  if( !data && !buffer )
    throw Error(InvalidState,"dsp::TimeSeries::get_data()",
		"Neither data nor buffer is defined");
  if( !data )
    return ((const unsigned char*)buffer);

  return ((const unsigned char*)data);
}

//! Return pointer to the specified data block
float* dsp::TimeSeries::get_datptr (unsigned ichan, unsigned ipol)
{
  return (float*)get_udatptr(ichan,ipol);
}

//! Return pointer to the specified data block
const float*
dsp::TimeSeries::get_datptr (unsigned ichan, unsigned ipol) const
{
  return (const float*)get_udatptr(ichan,ipol);
}

double dsp::TimeSeries::mean (unsigned ichan, unsigned ipol)
{
  if (get_ndim() != 1)
    throw Error (InvalidState, "dsp::TimeSeries::mean", "ndim != 1");

  float* data = get_datptr (ichan, ipol);
  double mean = 0.0;
  
  uint64 _ndat = get_ndat();

  for (unsigned idat=0; idat<_ndat; idat++)
    mean += data[idat];

  return mean/double(get_ndat());
}

void dsp::TimeSeries::copy_configuration (const TimeSeries* copy)
{
  Observation::operator = (*copy);
}

dsp::TimeSeries& dsp::TimeSeries::operator = (const TimeSeries& copy)
{
  data = copy.data;
  DataSeries::operator=(copy);
  return *this;

  /*
  if (this == &copy)
    return *this;

  Observation::operator=(copy);
  nbit = copy.get_nbit();

  resize (copy.get_ndat());

  uint64 npt = get_ndat() * get_ndim();

  //fprintf(stderr,"In dsp::TimeSeries::operator=() with ndat="UI64" nchan=%d get_nbytes()="UI64" npt="UI64"\n",
  //  get_ndat(),nchan,get_nbytes(),npt);
  //fprintf(stderr,"Got subsize="UI64" size="UI64" buffer=%p\n",
  //  subsize,size,buffer);
  //fprintf(stderr,"For copy got subsize="UI64" size="UI64" buffer=%p\n",
  //  copy.subsize,copy.size,copy.buffer);

  for (unsigned ichan=0; ichan<get_nchan(); ichan++){
    for (unsigned ipol=0; ipol<get_npol(); ipol++) {
      float* data1 = get_datptr (ichan, ipol);
      const float* data2 = copy.get_datptr (ichan, ipol);

      //char dummy[1024];
      //sprintf(dummy,"%p",data1);
      //string address = dummy;

      //if( address=="0x519db008" ){
      //fprintf(stderr,"broken data1=%p data2=%p\n",data1,data2);
      //exit(0);
      //}

      for (uint64 ipt=0; ipt<npt; ipt++)
        data1[ipt] = data2[ipt];
    }
  }
  

  return *this;
  */
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
  if( get_ndat()==0 )
    return;

  uint64 npt = get_ndat() * get_ndim();
  for (unsigned ichan=0; ichan<get_nchan(); ichan++){
    for (unsigned ipol=0; ipol<get_npol(); ipol++) {
      float* dat = get_datptr (ichan, ipol);
      for (uint64 ipt=0; ipt<npt; ipt++)
        dat[ipt]=0.0;
    }
  }

}

//! return value is number of timesamples actually appended.
//! If it is zero, then none were and we assume 'this' is full
//! If it is nonzero, but not equal to little->get_ndat(), then 'this' is full too
//! If it is equal to little->get_ndat(), it may/may not be full.
uint64 dsp::TimeSeries::append (const dsp::TimeSeries* little)
{
  if( verbose )
    fprintf(stderr,"In dsp::TimeSeries::append()\n");

  uint64 ncontain = get_ndat() * get_ndim();
  uint64 ncopy = little->get_ndat() * little->get_ndim();

  append_checks(ncontain,ncopy,little);
  if( ncontain==subsize_samples() )
    return 0;

  if ( get_ndat() == 0 ) {
    Observation::operator=( *little );
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

  if( verbose )
    fprintf(stderr,"Returning from dsp::TimeSeries::append() with "UI64"\n",
	    ncopy / uint64(get_ndim()));
	    
  return ncopy / uint64(get_ndim());
}

uint64 dsp::TimeSeries::append(const dsp::TimeSeries* little,unsigned ichan,unsigned ipol){
  if( verbose )
    fprintf(stderr,"In dsp::TimeSeries::append()\n");

  uint64 ncontain = get_ndat() * get_ndim();
  uint64 ncopy = little->get_ndat() * little->get_ndim();
  
  append_checks(ncontain,ncopy,little);
  if( ncontain==subsize_samples() )
    return 0;

  if ( get_ndat() == 0 ) {
    Observation::operator=( *little );
    set_nchan(1);
    set_npol(1);
    set_centre_frequency( little->get_centre_frequency( ichan ) );
    set_bandwidth( little->get_bandwidth()/little->get_nchan() );
    set_dc_centred( false );
    set_ndat (0);
  }    
  else if( !contiguous(*little,true,ichan,ipol) )
    throw Error (InvalidState, "dsp::TimeSeries::append()",
		 "next TimeSeries is not contiguous");
  
  const float* from = little->get_datptr (ichan,ipol);
  float* to = get_datptr(0,0) + ncontain;
  memcpy (to, from, ncopy*sizeof(float));
  
  set_ndat (get_ndat() + ncopy/little->get_ndim());

  if( verbose )
    fprintf(stderr,"Returning from dsp::TimeSeries::append() with "UI64"\n",
	    ncopy / uint64(get_ndim()));

  return ncopy / uint64(get_ndim());
}			       

void dsp::TimeSeries::append_checks(uint64& ncontain,uint64& ncopy,
				    const TimeSeries* little){
  if( verbose ){
    fprintf(stderr,"ncopy="UI64"\n",ncopy);
    fprintf(stderr,"ncontain="UI64"\n",ncontain);
    fprintf(stderr,"subsize (in samples)="UI64"\n",subsize_samples());
    fprintf(stderr,"nchan=%d npol=%d\n",nchan,npol);
    fprintf(stderr,"subsize="UI64"\n",subsize);
  }

  if ( subsize_samples() <= ncontain + ncopy ){
    ncopy = subsize_samples() - ncontain;
    fprintf(stderr,"dsp::TimeSeries::append()- this append will fill up the timeseries from ndat="UI64" with ncopy="UI64" to ndat="UI64".\n",
	    get_ndat(), ncopy, get_ndat()+ncopy/little->get_ndim());
  }
}

dsp::TimeSeries& dsp::TimeSeries::swap_data(dsp::TimeSeries& ts)
{
  DataSeries::swap_data( ts );
  float* tmp = data; data = ts.data; ts.data = tmp;
  //std::swap(data,ts.data);

  return *this;
}

void dsp::TimeSeries::check (float min, float max)
{
  if (get_detected()) {
    min *= fabs(min);
    max *= max;
  }

  min *= scale;
  max *= scale;

  uint64 _ndat = get_ndat();

  for (unsigned ichan=0; ichan<get_nchan(); ichan++)
    for (unsigned ipol=0; ipol<get_npol(); ipol++) {

      float* dat = get_datptr (ichan, ipol);

      for (uint64 idat=0; idat<_ndat; idat++)
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

  resize(0);
  data = _data.release();
  buffer = (unsigned char*)data;
}

//! Call this when you do not want to transfer ownership of the array
void dsp::TimeSeries::attach(float* _data){
  if( !_data )
    throw Error(InvalidState,"dsp::TimeSeries::attach()",
		"NULL ptr has been passed in- you haven't properly allocated it using 'new' before passing it into this method");

  resize(0);
  data = _data;
  buffer = (unsigned char*)data;
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

//! Calculates the mean and the std dev of the timeseries, removes the mean, and scales to sigma
void dsp::TimeSeries::normalise(){
  if( get_ndim()!=1 )
    throw Error(InvalidState,"dsp::TimeSeries::normalise()",
		"You can only give a real timeseries to this method- ie ndim!=1");

  for( unsigned ichan=0; ichan<nchan; ichan++){
    for( unsigned ipol=0; ipol<npol; ipol++){
      float* dat = get_datptr(ichan,ipol);

      float mmean = many_findmean(dat,dat+get_ndat());
      float ssigma = sqrt( many_findvar( dat, dat+get_ndat(), mmean) );

      register unsigned the_ndat = get_ndat();

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

//-------------------------------------------------------------------

dsp::TimeSeries& dsp::TimeSeriesPtr::operator * () const
{
  if(!ptr)
    throw Error(InvalidState,"dsp::TimeSeriesPtr::operator*()",
		"You have called operator*() when ptr is NULL");

  return *ptr;
}

dsp::TimeSeries* dsp::TimeSeriesPtr::operator -> () const 
{ 
  if(!ptr) 
    throw Error(InvalidState,"dsp::TimeSeriesPtr::operator*()",
		"You have called operator*() when ptr is NULL");

  return ptr;
}
        
dsp::TimeSeriesPtr& dsp::TimeSeriesPtr::operator=(const TimeSeriesPtr& tsp){
  ptr = tsp.ptr; 
  return *this; 
}

dsp::TimeSeriesPtr::TimeSeriesPtr(const TimeSeriesPtr& tsp){ 
  operator=(tsp);
}

dsp::TimeSeriesPtr::TimeSeriesPtr(TimeSeries* _ptr){ 
  ptr = _ptr;
}

dsp::TimeSeriesPtr::TimeSeriesPtr(){ 
  ptr = 0;
}

dsp::TimeSeriesPtr::~TimeSeriesPtr(){ }

void dsp::ChannelPtr::init(TimeSeries* _ts,unsigned _ichan, unsigned _ipol)
{
  ts=_ts;
  ichan=_ichan;
  ipol = _ipol;
  ptr=ts->get_datptr(ichan,ipol);
}

dsp::ChannelPtr& dsp::ChannelPtr::operator=(const ChannelPtr& c){
  ts=c.ts;
  ptr=c.ptr;
  ichan=c.ichan;
  ipol=c.ipol;
  return *this;
}

dsp::ChannelPtr::ChannelPtr()
{
  ts = 0;
  ptr = 0;
  ichan=ipol=0;
}

dsp::ChannelPtr::ChannelPtr(TimeSeries* _ts,unsigned _ichan, unsigned _ipol)
{
  init(_ts,_ichan,_ipol);
}

dsp::ChannelPtr::ChannelPtr(TimeSeriesPtr _ts,unsigned _ichan, unsigned _ipol)
{
  init(_ts.ptr,_ichan,_ipol);
}

dsp::ChannelPtr::ChannelPtr(const ChannelPtr& c){ 
  operator=(c);
}

dsp::ChannelPtr::~ChannelPtr(){ }
    
float& dsp::ChannelPtr::operator[](unsigned index){
  return ptr[index];
}

float dsp::ChannelPtr::get_centre_frequency(){
  return ts->get_centre_frequency(ichan);
}
