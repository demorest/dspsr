#include "environ.h"
#include "genutil.h"
#include "fsleep.h"
#include "minmax.h"
#include "Error.h"

#include "dsp/Observation.h"
#include "dsp/BitSeries.h"
#include "dsp/DataSeries.h"
#include "dsp/TimeSeries.h"

int dsp::DataSeries::instantiation_count = 0;
int64 dsp::DataSeries::memory_used = 0;

dsp::DataSeries::DataSeries() : Observation() {
  init();
}

void dsp::DataSeries::init(){
  instantiation_count++;

  Observation::init();
  
  buffer = NULL;
  size = 0;
  subsize = 0;
  nbit = 8 * sizeof(float);
}
  
dsp::DataSeries::DataSeries(const DataSeries& ms) {
  init();
  operator=(ms);
}

dsp::DataSeries::~DataSeries(){
  resize(0);
  instantiation_count--;
}

//! Enforces that ndat*ndim must be an integer number of bytes
void dsp::DataSeries::set_ndat(uint64 _ndat){
  if( _ndat*get_ndim()*get_nbit() % 8 )
    throw Error(InvalidParam,"dsp::DataSeries::set_ndat()",
		"You've tried to set an ndat ("UI64") that gives a non-integer number of bytes per pol/chan grouping",
		_ndat);
  Observation::set_ndat( _ndat );
}

//! Enforces that ndat*ndim must be an integer number of bytes
void dsp::DataSeries::set_ndim(uint64 _ndim){
  if( _ndim*get_ndat()*get_nbit() % 8 )
    throw Error(InvalidParam,"dsp::DataSeries::set_ndim()",
		"You've tried to set an ndim (%d) that gives a non-integer number of bytes per pol/chan grouping",
		_ndim);
  Observation::set_ndim( _ndim );
}

uint64 dsp::DataSeries::subsize_samples(){
  return (subsize*8) / (get_nbit()*get_ndim());
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
void dsp::DataSeries::resize (uint64 nsamples)
{
  if (verbose)
    cerr << "dsp::DataSeries::resize (" << nsamples << ")" << endl;

  // Number of bits needed to allocate a single pol/chan group
  uint64 nbits_required = nsamples * get_nbit() * get_ndim();

  if( verbose )
    fprintf(stderr,"dsp::DataSeries::resize() got nbits required per group="UI64"\n",
	    nbits_required);

  if( nbits_required%8 )  // 8 bits per byte
    throw Error(InvalidParam,"dsp::DataSeries::resize()",
		"Your nbit=%d and you've tried allocating a buffer of "UI64" samples."
		"  This is not an integer number of bytes",
		get_nbit(),nsamples);

  // Number of bytes needed to be allocated
  uint64 require = (nbits_required*npol*nchan)/8;

  if (verbose)
    cerr << "dsp::DataSeries::resize require uchar[" << require << "];"
      " have uchar[" << size << "]" << endl;

  if (!require || require > size) {
    if (buffer){
      delete [] buffer;
      buffer = 0;
      memory_used -= size;
    }
    size = subsize = 0;
  }

  set_ndat( nsamples );

  if (!require)
    return;

  if (size == 0) {
    buffer = new unsigned char[require];
    size = require;
    memory_used += size;
  }

  subsize = (get_ndim() * nsamples * get_nbit())/8;

  if( verbose )
    fprintf(stderr,"Returning from dsp::DataSeries::resize() with "UI64" bytes allocated.  subsize="UI64" bytes\n",size,subsize);
}

//! Returns a uchar pointer to the first piece of data
unsigned char* dsp::DataSeries::get_data(){
  return buffer;
}

//! Returns a uchar pointer to the first piece of data
const unsigned char* dsp::DataSeries::const_get_data() const{
  return buffer;
}

//! Return pointer to the specified data block
unsigned char* dsp::DataSeries::get_udatptr (unsigned ichan, unsigned ipol)
{
  /*
  char dummy[1024];
  sprintf(dummy,"%p",buffer);
  string address = dummy;

  if( address=="0x5a99c008" ){
    fprintf(stderr,"buffer=%p get_data()=%p ichan=%d npol=%d ipol=%d subsize="UI64" ret=%p\n",
	    buffer,get_data(),ichan,npol,ipol,subsize,
	    ((unsigned char*)get_data()) + (ichan*npol+ipol)*subsize);
  }
  */

  return ((unsigned char*)get_data()) + (ichan*npol+ipol)*subsize;
}

//! Return pointer to the specified data block
const unsigned char*
dsp::DataSeries::get_udatptr (unsigned ichan, unsigned ipol) const
{
  /*
  char dummy[1024];
  sprintf(dummy,"%p",buffer);
  string address = dummy;

  if( address=="0x5a99c008" ){
    fprintf(stderr,"buffer=%p get_data()=%p ichan=%d npol=%d ipol=%d subsize="UI64" ret=%p\n",
	    buffer,get_data(),ichan,npol,ipol,subsize,
	    ((unsigned char*)get_data()) + (ichan*npol+ipol)*subsize);
  }
  */

  return ((const unsigned char*)const_get_data()) + (ichan*npol+ipol)*subsize;
}

dsp::DataSeries& dsp::DataSeries::operator = (const DataSeries& copy)
{
  if (this == &copy)
    return *this;

  Observation::operator = (copy);

  resize (copy.get_ndat());

  uint64 npt = (get_ndat() * get_ndim() * get_nbit())/8;

  /*
  fprintf(stderr,"In dsp::DataSeries::operator=() with ndat="UI64" nchan=%d get_nbytes()="UI64" npt="UI64"\n",
	  get_ndat(),nchan,get_nbytes(),npt);

  fprintf(stderr,"Got subsize="UI64" size="UI64" buffer=%p\n",
	  subsize,size,buffer);
  fprintf(stderr,"For copy got subsize="UI64" size="UI64" buffer=%p\n",
	  copy.subsize,copy.size,copy.buffer);
  */

  for (unsigned ichan=0; ichan<get_nchan(); ichan++){
    for (unsigned ipol=0; ipol<get_npol(); ipol++) {
      //  fprintf(stderr,"hi1\n");

      unsigned char* dest = get_udatptr (ichan, ipol);
      const unsigned char* src = copy.get_udatptr (ichan, ipol);
      //float* dest = dynamic_cast<TimeSeries*>(this)->get_datptr (ichan, ipol);
      //const float* src = dynamic_cast<TimeSeries*>(const_cast<DataSeries*>(&copy))->get_datptr (ichan, ipol);

      /*
      fprintf(stderr,"Got dest=%p src=%p npt="UI64"\n",dest,src,npt);
      fprintf(stderr,"hi2\n");

      char dummy[1024];
      sprintf(dummy,"%p",dest);
      string address = dummy;

      if( address=="0x519db008" ){
	fprintf(stderr,"not broken!!!\n");
	fprintf(stderr,"src=%p dest=%p\n",src,dest);
	fprintf(stderr,"not broken!!!\n");
	
	DataSeries* bla = (DataSeries*)&copy;

	fprintf(stderr,"nonconst: %p const: %p\n",
		bla->get_udatptr(ichan,ipol),src);
	//exit(0);
      }
          

      for( unsigned i=0; i<npt; i++){
	if( address=="0x519db008" ){
	  fprintf(stderr,"i=%d src[0]=%f\n",i,i,*((float*)src));
	}
	dest[i] = src[i];
      }
      */
      memcpy(dest,src,npt);

      //fprintf(stderr,"hi3\n");
    }
  }
  //  fprintf(stderr,"hi4\n");
  
  return *this;
}

dsp::DataSeries& dsp::DataSeries::swap_data(dsp::DataSeries& ts)
{
  Observation::swap_data( ts );
  unsigned char* tmp = buffer; buffer = ts.buffer; ts.buffer = tmp;
  //  std::swap(buffer,ts.buffer);
  
  uint64 tmp2 = size; size = ts.size; ts.size = tmp2;
  //  std::swap(size,ts.size);

  uint64 tmp3 = subsize; subsize = ts.subsize; ts.subsize = tmp3;
  //  std::swap(subsize,ts.subsize);

  return *this;
}
