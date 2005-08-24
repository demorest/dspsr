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
  DataSeries::set_nbit( 8 * sizeof(float) );
  data = 0;  
  reserve_ndat = 0;
  set_preserve_seeked_data( false );
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
  if( verbose )
    fprintf(stderr,"In dsp::TimeSeries::resize() with data=%p and buffer=%p\n",
	    data,buffer);

  if( data && buffer && verbose ){
    cerr << "dsp::TimeSeries::resize(" << nsamples << ") offset="
         << int64((data-(float*)buffer)) << endl;
    cerr << "get_samps_offset() = " 
         << get_samps_offset() << " floats get_preserve_seeked_data=" << get_preserve_seeked_data() << endl;
  }

  if( !get_preserve_seeked_data() ){
    if( verbose ) 
      cerr << "dsp::TimeSeries::resize not preserving; reserving "
           << reserve_ndat << endl;
    DataSeries::resize(nsamples+reserve_ndat);
    data = (float*)buffer + reserve_ndat * get_ndim();
    return;
  }

  uint64 samps_offset = get_samps_offset();
  unsigned char* old_buffer = 0;
  uint64 old_size = size;
  MJD true_start_time = get_start_time();
  
  if( subsize*get_npol()*get_nchan() > size )
    throw Error(InvalidState,"dsp::TimeSeries::resize()",
		"BUG! Cannot calculate startings properly as subsize*get_npol()*get_nchan() > size ("UI64" * %d * %d > "UI64")\n",
		subsize,get_npol(),get_nchan(),size);

  vector<vector<float*> > startings(get_nchan(),vector<float*>(get_npol()));

  for( unsigned ichan=0; ichan<get_nchan(); ichan++){
    for( unsigned ipol=0; ipol<get_npol(); ipol++){
      startings[ichan][ipol] = get_datptr(ichan,ipol) - get_samps_offset()*get_ndim();
      //      fprintf(stderr,"startings(%d,%d) is "I64" samps past buffer size="UI64" subsize="UI64" ndat="UI64" npol=%d ndim=%d nchan=%d\n",
      //      ichan,ipol,int64(startings[ichan][ipol]-(float*)buffer),size,subsize,get_ndat(),
      //      get_npol(),get_ndim(),get_nchan());
    }
  }

  DataSeries::resize(0,old_buffer);
  DataSeries::resize(nsamples+samps_offset);

  data = (float*)buffer;

  for( unsigned ichan=0; ichan<get_nchan(); ichan++){
    for( unsigned ipol=0; ipol<get_npol(); ipol++){
      float* dat = get_datptr(ichan,ipol);
      float* src = startings[ichan][ipol];
      memcpy( dat, src, samps_offset*get_ndim()*sizeof(float));
    }
  }

  delete [] old_buffer;
  memory_used -= old_size;

  seek( samps_offset );
  set_start_time( true_start_time );

  //  fprintf(stderr,"Bye from dsp::TimeSeries::resize("UI64") with offset="I64" floats and first val=%f\n",
  //  nsamples,int64((data-(float*)buffer)),
  //  get_datptr(0,0)[0]);

  if( data && buffer && verbose )
    fprintf(stderr,"Bye from dsp::TimeSeries::resize("UI64") with offset="I64" or "I64" floats\n",
	    nsamples,int64((data-(float*)buffer)),get_samps_offset());
}

dsp::TimeSeries& dsp::TimeSeries::operator = (const TimeSeries& copy)
{
  //  fprintf(stderr,"Hi from dsp::TimeSeries::operator=("UI64") with offset="I64"\n",
  //  copy.get_ndat(),int64((data-(float*)buffer)*get_ndim()));

  if( !get_preserve_seeked_data() ){
    //fprintf(stderr,"dsp::TimeSeries::operator=() won't preserve data\n");
    DataSeries::operator=(copy);
    data = (float*)buffer;
    //fprintf(stderr,"dsp::TimeSeries::operator=() returning\n");
    return *this;
  }

  //  fprintf(stderr,"dsp::TimeSeries::operator=() will preserve data... calling resize()\n");
  resize( copy.get_ndat() );
  set_ndat( 0 );
  //  fprintf(stderr,"dsp::TimeSeries::operator=()... calling append()\n");
  append( &copy );

  //  fprintf(stderr,"Bye from dsp::TimeSeries::operator=("UI64") with offset="I64"\n",
  //	  copy.get_ndat(),int64((data-(float*)buffer)*get_ndim()));

  return *this;
}

//! Use the supplied array to store nsamples time samples.
//! Always deletes any existing data
void dsp::TimeSeries::resize( uint64 nsamples, uint64 bytes_supplied, unsigned char* buffer_supplied){
  uint64 require = uint64(get_ndim()) * nsamples * uint64(get_npol()) * uint64(get_nchan());

  if( require > bytes_supplied )
    throw Error(InvalidState,"dsp::TimeSeries::resize()",
		"Not enough bytes supplied ("UI64") to fit in "UI64" samples -> "UI64" bytes",
		bytes_supplied, nsamples, require );

  if( get_preserve_seeked_data() )
    throw Error(InvalidState,"dsp::TimeSeries::resize(uint64,uint64,uchar*)",
		"You have preserved_seeked_data set to true, but this method is not programmed to do any preserving of seeked data");

  DataSeries::resize(0);

  buffer = buffer_supplied;
  data = (float*)buffer_supplied;
  size = bytes_supplied;
  set_ndat( nsamples );
  subsize = (get_ndim() * nsamples * get_nbit())/8;

  if( subsize*get_npol()*get_nchan() > size )
    throw Error(InvalidState,"dsp::TimeSeries::resize(uint64,uint64,uchar*)",
		"BUG! subsize*get_npol()*get_nchan() > size ("UI64" * %d * %d > "UI64")\n",
		subsize,get_npol(),get_nchan(),size);
}

//! Equivalent to resize(0) but instead of deleting data, returns the pointer for reuse elsewhere
void dsp::TimeSeries::zero_resize(unsigned char*& _buffer, uint64& nbytes){
  if( get_preserve_seeked_data() )
    throw Error(InvalidState,"dsp::TimeSeries::zero_resize()",
		"You have preserved_seeked_data set to true, but this method is not programmed to do any preserving of seeked data");

  _buffer = (unsigned char*)buffer;
  nbytes = size;

  buffer = 0;
  data = 0;
  DataSeries::resize(0);
}

//! Offset the base pointer by offset time samples
void dsp::TimeSeries::seek (int64 offset)
{
  if (offset == 0)
    return;

  if (offset > int64(get_ndat()))
    throw Error (InvalidRange, "dsp::TimeSeries::seek",
		 "offset="I64" > ndat="UI64";"
                 " attempt to seek past end of data",
                 offset, get_ndat());

  int64 current_offset = get_seekage();

  if (-offset > current_offset)
    throw Error (InvalidRange, "dsp::TimeSeries::seek",
		 "offset="I64" > current_offset="I64";"
                 " attempt to seek before start of data", 
                 offset, current_offset);

  data += offset * int64(get_ndim());
  set_ndat( get_ndat() - offset );

  change_start_time (offset);
}

//! Returns a uchar pointer to the first piece of data
unsigned char* dsp::TimeSeries::get_data(){
  if( !data && !buffer )
    throw Error(InvalidState,"dsp::TimeSeries::get_data()",
		"Neither data nor buffer is defined.  ndat="UI64,
		get_ndat());
  if( !data )
    data = (float*)buffer;
  return ((unsigned char*)data);
}

//! Returns a uchar pointer to the first piece of data
const unsigned char* dsp::TimeSeries::get_data() const{
  if( !data && !buffer )
    throw Error(InvalidState,"dsp::TimeSeries::const_get_data()",
		"Neither data nor buffer is defined.  ndat="UI64,
		get_ndat());
  if( !data )
    return ((const unsigned char*)buffer);
  return ((const unsigned char*)data);
}

//! Returns a uchar pointer to the first piece of data
const unsigned char* dsp::TimeSeries::const_get_data() const{
  if( !data && !buffer )
    throw Error(InvalidState,"dsp::TimeSeries::const_get_data()",
		"Neither data nor buffer is defined.  ndat="UI64,
		get_ndat());
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

void dsp::TimeSeries::copy_configuration (const Observation* copy){
  if( copy==this )
    return;

  if (!get_preserve_seeked_data() || get_samps_offset() == 0) {
    Observation::operator=( *copy );
    return;
  }

  Observation* input = (Observation*)copy;
  
  unsigned input_nchan = input->get_nchan();
  unsigned input_npol = input->get_npol();
  unsigned input_ndim = input->get_ndim();
  Signal::State input_state = input->get_state();

  input->set_npol( get_npol() );
  input->set_ndim( get_ndim() );
  input->set_state( get_state() );
  input->set_nchan( get_nchan() );

  Observation::operator=( *input );

  input->set_nchan( input_nchan );
  input->set_npol( input_npol );
  input->set_ndim( input_ndim );
  input->set_state( input_state );

  if (verbose)
    cerr << "dsp::TimeSeries::copy_configuration"
      " not copying nchan, npol, ndim, state" << endl;
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


void dsp::TimeSeries::prepend (const dsp::TimeSeries* pre, uint64 pre_ndat)
{
  if (!pre)
    return;

  if (!pre_ndat)
    pre_ndat = pre->get_ndat();

  seek (-int64(pre_ndat));

  copy_data (pre, 0, pre_ndat);
}

void dsp::TimeSeries::copy_data (const dsp::TimeSeries* copy, 
				 uint64 idat_start, uint64 copy_ndat)
{
  if (!copy_ndat || !copy)
    return;

  uint64 offset = idat_start * get_ndim();
  uint64 byte_count = copy_ndat * get_ndim() * sizeof(float);

  for (unsigned ichan=0; ichan<get_nchan(); ichan++) {
    for (unsigned ipol=0; ipol<get_npol(); ipol++) {
      float* to = get_datptr (ichan, ipol);
      const float* from = copy->get_datptr(ichan,ipol) + offset;
      memcpy (to, from, byte_count);
    }
  }
}

/*! 
  \retval number of timesamples actually appended.  
  If zero, then none were and we assume 'this' is full.
  If nonzero, but not equal to little->get_ndat(), then 'this' is full too.
  If equal to little->get_ndat(), it may/may not be full.
*/
uint64 dsp::TimeSeries::append (const dsp::TimeSeries* little)
{
  if( verbose )
    fprintf(stderr,"In dsp::TimeSeries::append()\n");

  uint64 ncontain = get_ndat();
  uint64 ncopy = little->get_ndat();

  append_checks(ncontain,ncopy,little);
  if( ncontain==maximum_ndat() )
    return 0;

  if ( get_ndat() == 0 ) {
    int64 samps_offset = get_samps_offset();

    if( get_preserve_seeked_data() && samps_offset > 0 ){
      seek( -samps_offset );
      uint64 ret = append( little );
      seek( samps_offset );
      return ret;
    }
    Observation::operator=( *little );
    set_ndat (0);
  }    
  else if( !contiguous(*little,true) )
    throw Error (InvalidState, "dsp::TimeSeries::append()",
		 "next TimeSeries is not contiguous");
  
  for( unsigned ichan=0; ichan<get_nchan(); ichan++) {
    for( unsigned ipol=0; ipol<get_npol(); ipol++) {
      const float* from = little->get_datptr (ichan,ipol);
      float* to = get_datptr(ichan,ipol) + ncontain*get_ndim();
      
      memcpy (to, from, ncopy*get_ndim()*sizeof(float));
    }
  }

  set_ndat (ncontain + ncopy);

  if( verbose )
    fprintf(stderr,"Returning from dsp::TimeSeries::append() with "UI64"\n",
	    ncopy);
	    
  return ncopy;
}

uint64 dsp::TimeSeries::append(const dsp::TimeSeries* little,unsigned ichan,unsigned ipol){
  if( verbose )
    fprintf(stderr,"In dsp::TimeSeries::append()\n");

  uint64 ncontain = get_ndat();
  uint64 ncopy = little->get_ndat();
  
  append_checks(ncontain,ncopy,little);
  if( ncontain==maximum_ndat() )
    return 0;

  if ( get_ndat() == 0 ) {
    int64 samps_offset = get_samps_offset();

    if( get_preserve_seeked_data() && samps_offset > 0 ){
      seek( -samps_offset );
      uint64 ret = append( little );
      seek( samps_offset );
      return ret;
    }

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
  float* to = get_datptr(0,0) + ncontain*get_ndim();
  memcpy (to, from, ncopy*get_ndim()*sizeof(float));
  
  set_ndat (ncontain + ncopy);

  if( verbose )
    fprintf(stderr,"Returning from dsp::TimeSeries::append() with "UI64"\n",
	    ncopy);

  return ncopy;
}			       

void dsp::TimeSeries::append_checks(uint64& ncontain,uint64& ncopy,
				    const TimeSeries* little){
  if( verbose ){
    fprintf(stderr,"dsp::TimeSeries::append_checks() ncopy="UI64"\n",ncopy);
    fprintf(stderr,"dsp::TimeSeries::append_checks() ncontain="UI64"\n",ncontain);
    fprintf(stderr,"dsp::TimeSeries::append_checks() maximum_ndat()="UI64"\n",maximum_ndat());
    fprintf(stderr,"dsp::TimeSeries::append_checks() nchan=%d npol=%d\n",get_nchan(),get_npol());
    fprintf(stderr,"dsp::TimeSeries::append_checks() subsize="UI64"\n",subsize);
  }

  if ( maximum_ndat() <= ncontain + ncopy ){
    ncopy = maximum_ndat() - ncontain;
    if( verbose )
      fprintf(stderr,"dsp::TimeSeries::append()- this append will fill up the timeseries from ndat="UI64" with ncopy="UI64" to ndat="UI64".\n",
	      get_ndat(), ncopy, get_ndat()+ncopy/little->get_ndim());
  }
}

dsp::TimeSeries& dsp::TimeSeries::swap_data(dsp::TimeSeries& ts)
{
  if( get_preserve_seeked_data() )
    throw Error(InvalidState,"dsp::TimeSeries::swap_data()",
		"This method is not programmed to handle cases when preserve_seeked_data is set to true");

  DataSeries::swap_data( ts );
  float* tmp = data; data = ts.data; ts.data = tmp;

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
/*! This is dangerous as it ASSUMES new data buffer has been
 pre-allocated and is big enough.  Beware of segmentation faults when
 using this routine.  Also do not try to delete the old memory once
 you have called this- the TimeSeries::data member now owns it. */
void dsp::TimeSeries::attach (auto_ptr<float> _data)
{
  if( !_data.get() )
    throw Error(InvalidState,"dsp::TimeSeries::attach()",
		"NULL auto_ptr has been passed in- you haven't properly allocated it using 'new' before passing it into this method");
  if( get_preserve_seeked_data() )
    throw Error(InvalidState,"dsp::TimeSeries::attach(float*)",
		"attach() isn't programmed to handle situations when preserve_seeked_data is set to true");

  resize(0);
  data = _data.release();
  buffer = (unsigned char*)data;
}

//! Call this when you do not want to transfer ownership of the array
void dsp::TimeSeries::attach(float* _data){
  if( !_data )
    throw Error(InvalidState,"dsp::TimeSeries::attach()",
		"NULL ptr has been passed in- you haven't properly allocated it using 'new' before passing it into this method");
  if( get_preserve_seeked_data() )
    throw Error(InvalidState,"dsp::TimeSeries::attach(float*)",
		"attach() isn't programmed to handle situations when preserve_seeked_data is set to true");

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

  for( unsigned ichan=0; ichan<get_nchan(); ichan++){
    for( unsigned ipol=0; ipol<get_npol(); ipol++){
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

//! Over-rides DataSeries::set_nchan()- this only allows a change if preserve_seeked_data is false
void dsp::TimeSeries::set_nchan(unsigned _nchan){
  if( get_preserve_seeked_data() && get_samps_offset() != 0 && _nchan != get_nchan() )
    throw Error(InvalidParam,"dsp::TimeSeries::set_nchan()",
		"You have preserve_seeked_data set to true with "UI64" samples being preserved but you want to change nchan from %d to %d.  This is currently not programmed for",
		get_samps_offset(),get_nchan(),_nchan);

  DataSeries::set_nchan(_nchan);
}

//! Over-rides DataSeries::set_npol()- this only allows a change if preserve_seeked_data is false
void dsp::TimeSeries::set_npol(unsigned _npol){
  if( get_preserve_seeked_data() && get_samps_offset() != 0 && _npol != get_npol() )
    throw Error(InvalidParam,"dsp::TimeSeries::set_npol()",
		"You have preserve_seeked_data set to true with "UI64" samples being preserved but you want to change npol from %d to %d.  This is currently not programmed for",
		get_samps_offset(),get_npol(),_npol);

  DataSeries::set_npol(_npol);
}

//! Over-rides DataSeries::set_npol()- this only allows a change if preserve_seeked_data is false
void dsp::TimeSeries::set_ndim(unsigned _ndim){
  if( get_preserve_seeked_data() && get_samps_offset() != 0 && _ndim != get_ndim() )
    throw Error(InvalidParam,"dsp::TimeSeries::set_ndim()",
		"You have preserve_seeked_data set to true with "UI64" samples being preserved but you want to change ndim from %d to %d.  This is currently not programmed for",
		get_samps_offset(),get_ndim(),_ndim);

  DataSeries::set_ndim(_ndim);
}

/*! This method is used by the InputBuffering policy */
void dsp::TimeSeries::change_reserve (int64 change)
{
  if (verbose)
    cerr << "dsp::TimeSeries::change_reserve (" << change << ")" << endl;

  if (change < 0) {
    uint64 decrease = -change;
    if (decrease > reserve_ndat)
    throw Error (InvalidState, "dsp::TimeSeries::change_reserve",
		 "decrease="I64"; reserve_ndat="UI64, 
		 decrease, reserve_ndat);

    reserve_ndat -= decrease;
  }
  else
    reserve_ndat += change;
}

//! Returns how many samples have been seeked over
uint64 dsp::TimeSeries::get_seekage(){     
  float* fbuffer = (float*)buffer;
  
  return uint64(data - fbuffer) / uint64(get_ndim());
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
