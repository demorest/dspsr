#include <assert.h>
#include <stdio.h>
#include <malloc.h>

#include "dsp/DataSeries.h"
#include "Error.h"

int dsp::DataSeries::instantiation_count = 0;
int64 dsp::DataSeries::memory_used = 0;

dsp::DataSeries::DataSeries() : Observation() {
  initi();
}

void dsp::DataSeries::initi(){
  instantiation_count++;
  //  fprintf(stderr,"New dataseries: %p count=%d\n", this,
  //  instantiation_count);

  Observation::init();
  
  buffer = NULL;
  size = 0;
  subsize = 0;
  set_nbit( 8 * sizeof(float) );
}
  
dsp::DataSeries::DataSeries(const DataSeries& ms) {
  initi();
  operator=(ms);
}

dsp::DataSeries::~DataSeries(){
  //  fprintf(stderr,"In DataSeries destructor ndat="UI64" %p\n",get_ndat(), this);
  resize(0);
  instantiation_count--;
  //  fprintf(stderr,"dsp::DataSeries::~DataSeries() count now %d\n",
  //  instantiation_count);
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
		"You've tried to set an ndim ("UI64") that gives a non-integer number of bytes per pol/chan grouping",
		_ndim);
  Observation::set_ndim( unsigned(_ndim) );
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
void dsp::DataSeries::resize (uint64 nsamples){
  unsigned char* dummy = (unsigned char*)(-1);
  resize(nsamples,dummy);
}

#define INTERACTIVE_MEMORY 0

void dsp::DataSeries::resize (uint64 nsamples, unsigned char*& old_buffer)
{
  if (verbose)
    cerr << "dsp::DataSeries::resize (" << nsamples << ") nbit="
         << get_nbit() << " ndim=" << get_ndim()
	 << " ndat=" << get_ndat() << endl;

  // Number of bits needed to allocate a single pol/chan group
  uint64 nbits_required = nsamples * get_nbit() * get_ndim();

  if (verbose)
    cerr << "dsp::DataSeries::resize nbits=" << nbits_required << endl;

  if (nbits_required & 0x07)  // 8 bits per byte
    throw Error (InvalidParam,"dsp::DataSeries::resize",
		"nbit=%d ndim=%d nsamp="UI64" not an integer number of bytes",
		get_nbit(), get_ndim(), nsamples);

  // Number of bytes needed to be allocated
  uint64 require = (nbits_required*get_npol()*get_nchan())/8;

  if (verbose)
    cerr << "dsp::DataSeries::resize require uchar[" << require << "];"
      " have uchar[" << size << "]" << endl;

  if (!require || require > size) {
    if (buffer){
      if( old_buffer != (unsigned char*)(-1) ){
	old_buffer = buffer;
      }
      else{
#if INTERACTIVE_MEMORY
	cerr << "dsp::DataSeries::resize free " << size << " bytes at "
	     << (void*)buffer << endl;
	getchar();
#endif
	free (buffer);
#if INTERACTIVE_MEMORY
	cerr << "dsp::DataSeries::resize freed" << endl;
#endif
	memory_used -= size;
      }
      buffer = 0;
    }
    size = subsize = 0;
  }

  set_ndat( nsamples );

  if (!require)
    return;

  if (size == 0) {

    // Add '8' (2 floats) on for FFTs that require 2 extra floats in the allocated memory

#if INTERACTIVE_MEMORY
    cerr << "dsp::DataSeries::resize allocate " << require << " bytes" << endl;
    getchar();
#endif
    buffer = (unsigned char*) memalign (16, size_t(require + 8));
#if INTERACTIVE_MEMORY
    cerr << "dsp::DataSeries::resize allocated at " << (void*) buffer << endl;
#endif

    if( !buffer )
      throw Error(InvalidState,"dsp::DataSeries::resize()",
		  "Could not allocate another "UI64" bytes!",
		  require+8);

    size = require;
    memory_used += size;
  }

  reshape ();
}

void dsp::DataSeries::reshape ()
{
  subsize = (get_ndim() * get_ndat() * get_nbit()) / 8;
  
  if (subsize*get_npol()*get_nchan() > size)
    throw Error (InvalidState, "dsp::DataSeries::reshape",
		 "subsize="UI64" * npol=%d * nchan=%d > size="UI64,
		 subsize, get_npol(), get_nchan(), size);

  if (verbose)
    cerr << "dsp::DataSeries::reshape size=" << size << " bytes"
      " (subsize=" << subsize << " bytes)" << endl;
}

void dsp::DataSeries::reshape (unsigned _npol, unsigned _ndim)
{
  unsigned _total = _npol * _ndim;
  unsigned total = get_npol() * get_ndim();

  if (total != _total)
    throw Error (InvalidParam, "dsp::DataSeries::reshape",
		 "current npol=%d*ndim=%d = %d != new npol=%d*ndim=%d = %d",
		 get_npol(), get_ndim(), total, _npol, _ndim, _total);

  subsize *= get_npol();
  subsize /= _npol;

  set_npol (_npol);
  set_ndim (_ndim);
}

//! Returns a uchar pointer to the first piece of data
unsigned char* dsp::DataSeries::get_data()
{
  return buffer;
}

//! Returns a uchar pointer to the first piece of data
const unsigned char* dsp::DataSeries::get_data() const
{
  return buffer;
}

//! Return pointer to the specified data block
unsigned char* dsp::DataSeries::get_udatptr (unsigned ichan, unsigned ipol)
{
  if( ichan >= get_nchan() )
    throw Error(InvalidState," dsp::DataSeries::get_udatptr()",
		"Your ichan (%d) was >= nchan (%d)",
		ichan,get_nchan());
  if( ipol >= get_npol() )
    throw Error(InvalidState," dsp::DataSeries::get_udatptr()",
		"Your ipol (%d) was >= npol (%d)",
		ipol,get_npol()); 

  return get_data() + (ichan*get_npol()+ipol)*subsize;
}

//! Return pointer to the specified data block
const unsigned char*
dsp::DataSeries::get_udatptr (unsigned ichan, unsigned ipol) const
{
  if( ichan >= get_nchan() )
    throw Error(InvalidState," dsp::DataSeries::get_udatptr()",
		"Your ichan (%d) was >= nchan (%d)",
		ichan,get_nchan());
  if( ipol >= get_npol() )
    throw Error(InvalidState," dsp::DataSeries::get_udatptr()",
		"Your ipol (%d) was >= npol (%d)",
		ipol,get_npol()); 

  return get_data() + (ichan*get_npol()+ipol)*subsize;
}

dsp::DataSeries& dsp::DataSeries::operator = (const DataSeries& copy)
{
  //  fprintf(stderr,"Entered dsp::DataSeries::operator =()\n");

  if (this == &copy)
    return *this;

  Observation::operator = (copy);

  resize (copy.get_ndat());

  uint64 npt = (get_ndat() * get_ndim() * get_nbit())/8;

  for (unsigned ichan=0; ichan<get_nchan(); ichan++){
    for (unsigned ipol=0; ipol<get_npol(); ipol++) {
      unsigned char* dest = get_udatptr (ichan, ipol);
      const unsigned char* src = copy.get_udatptr (ichan, ipol);
      memcpy(dest,src,size_t(npt));
    }
  }
  
  //  fprintf(stderr,"Returning from dsp::DataSeries::operator =()\n");

  return *this;
}

dsp::DataSeries& dsp::DataSeries::swap_data(dsp::DataSeries& ts)
{
  Observation::swap_data( ts );
  unsigned char* tmp = buffer; buffer = ts.buffer; ts.buffer = tmp;
  uint64 tmp2 = size; size = ts.size; ts.size = tmp2;
  uint64 tmp3 = subsize; subsize = ts.subsize; ts.subsize = tmp3;

  if( subsize*get_npol()*get_nchan() > size )
    throw Error(InvalidState,"dsp::DataSeries::swap_data()",
		"BUG! subsize*get_npol()*get_nchan() > size ("UI64" * %d * %d > "UI64")\n",
		subsize,get_npol(),get_nchan(),size);

  return *this;
}

//! Returns the number of samples that have been seeked over
int64 dsp::DataSeries::get_samps_offset() const {
  uint64 bytes_offset = get_data() - buffer; 
  uint64 samps_offset = (bytes_offset * 8)/(get_nbit()*get_ndim());
  //  fprintf(stderr,"\ndsp::DataSeries::get_samps_offset() got bytes_offset="UI64" nbit=%d ndim=%d so samp offset="UI64"\n",
  //  bytes_offset,get_nbit(),get_ndim(),samps_offset);
  return int64(samps_offset);
}

//! Returns the maximum ndat possible with current offset of data from base pointer
uint64 dsp::DataSeries::maximum_ndat() const {
  uint64 bytes_offset = get_data() - buffer; 
  uint64 bits_avail = (subsize-bytes_offset) * 8;
  //  fprintf(stderr,"dsp::DataSeries::maximum_ndat() got bytes_offset="UI64" subsize="UI64" bits_avail="UI64" returning "UI64"/(%d*%d)="UI64"\n",
  //  bytes_offset,subsize,bits_avail,bits_avail,get_ndim(),get_nbit(),bits_avail/(get_ndim()*get_nbit()));
  return bits_avail/(get_ndim()*get_nbit());
}

//! Checks that ndat is not too big for size and subsize
void dsp::DataSeries::check_sanity() const {
  if( get_nbytes(get_ndat()+get_samps_offset()) > size )
    throw Error(InvalidState,"dsp::DataSeries::check_sanity()",
		"ndat="UI64" samps_offset="I64" nbytes used="UI64" is greater than bytes allocated ("UI64")",
		get_ndat(), get_samps_offset(),
		get_nbytes(get_ndat()+get_samps_offset()),
		size);

  if( (get_nbit()*get_ndim()*(get_ndat()+get_samps_offset()))/8 > subsize )
    throw Error(InvalidState,"dsp::DataSeries::check_sanity()",
		"ndat="UI64" samps_offset="I64" bytes used in a block="UI64" is greater than subsize="UI64,
		get_ndat(), get_samps_offset(),
		(get_nbit()*get_ndim()*(get_ndat()+get_samps_offset()))/8,
		subsize);
}
