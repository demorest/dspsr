/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DataSeries.h"
#include "dsp/Memory.h"

#include "Error.h"

#include <string.h>

using namespace std;

int dsp::DataSeries::instantiation_count = 0;
int64_t dsp::DataSeries::memory_used = 0;

dsp::DataSeries::DataSeries() : Observation()
{
  initi();
}

void dsp::DataSeries::initi()
{
  instantiation_count++;

  Observation::init();
  
  memory = Memory::get_manager();

  buffer = NULL;
  size = 0;
  subsize = 0;
  set_nbit( 8 * sizeof(float) );
}
  
dsp::DataSeries::DataSeries (const DataSeries& ms)
{
  initi();
  operator=(ms);
}

dsp::DataSeries::~DataSeries()
{
  resize(0);
  instantiation_count--;
}

void dsp::DataSeries::set_memory (Memory* m)
{
  if (memory && buffer)
    resize (0);

  memory = m;
}

const dsp::Memory* dsp::DataSeries::get_memory () const
{
  return memory;
}

//! Enforces that ndat*ndim must be an integer number of bytes
void dsp::DataSeries::set_ndat (uint64_t _ndat)
{
  if( _ndat*get_ndim()*get_nbit() % 8 )
    throw Error(InvalidParam,"dsp::DataSeries::set_ndat",
		"ndat="UI64" * ndim=%d * nbit=%d yields non-integer bytes",
                _ndat, get_ndim(), get_nbit());

  Observation::set_ndat( _ndat );
}

//! Enforces that ndat*ndim must be an integer number of bytes
void dsp::DataSeries::set_ndim (uint64_t _ndim)
{
  if( _ndim*get_ndat()*get_nbit() % 8 )
    throw Error(InvalidParam,"dsp::DataSeries::set_ndim",
                "ndat="UI64" * ndim=%d * nbit=%d yields non-integer bytes",
                get_ndat(), _ndim, get_nbit()); 

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
void dsp::DataSeries::resize (uint64_t nsamples)
{
  unsigned char* dummy = (unsigned char*)(-1);
  resize(nsamples,dummy);
}

#define INTERACTIVE_MEMORY 0

void dsp::DataSeries::resize (uint64_t nsamples, unsigned char*& old_buffer)
{
  if (verbose)
    cerr << "dsp::DataSeries::resize"
            " nsamp=" << nsamples << 
            " nbit=" << get_nbit() <<
            " ndim=" << get_ndim() << 
            " (current ndat=" << get_ndat() << ")" << endl;

  // Number of bits needed to allocate a single pol/chan group
  uint64_t nbits_required = nsamples * get_nbit() * get_ndim();

  if (verbose)
    cerr << "dsp::DataSeries::resize nbits=nsamp*nbit*ndim=" 
	 << nbits_required << endl;

  // check that nbits is a multiple of 8 (bits per byte)
  if (nbits_required & 0x07)
    throw Error (InvalidParam,"dsp::DataSeries::resize",
		"nbit=%d ndim=%d nsamp="UI64" not an integer number of bytes",
		get_nbit(), get_ndim(), nsamples);

  if (verbose)
    cerr << "dsp::DataSeries::resize"
            " npol=" << get_npol() << 
            " nchan=" << get_nchan() << endl;

  // Number of bytes needed to be allocated
  uint64_t require = (nbits_required*get_npol()*get_nchan())/8;

  if (verbose)
    cerr << "dsp::DataSeries::resize nbytes=nbits/8*npol*nchan=" << require 
         << " (current size=" << size << ")" << endl;

  if (!require || require > size) {
    if (buffer){
      if( old_buffer != (unsigned char*)(-1) )
      {
	old_buffer = buffer;
      }
      else
      {
	if (verbose)
          cerr << "dsp::DataSeries::resize Memory::free size=" << size << " buffer="
	       << (void*)buffer << endl;
	memory->do_free (buffer);
	memory_used -= size;
      }
      buffer = 0;
    }
    size = subsize = 0;
  }

  set_ndat( nsamples );

  if (!require)
    return;

  if (size == 0)
  {
    if (verbose)
      cerr << "dsp::DataSeries::resize Memory::allocate (" << require << ")" << endl;
    buffer = (unsigned char*) memory->do_allocate (require);

    if (verbose)
      cerr << "dsp::DataSeries::resize buffer=" << (void*) buffer << endl;

    if (!buffer)
      throw Error (InvalidState,"dsp::DataSeries::resize",
		  "Could not allocate "UI64" bytes", require);

    size = require;
    memory_used += size;

    if (verbose)
      cerr << "dsp::DataSeries::resize memory_used=" << memory_used << endl;
  }

  reshape ();
}

void dsp::DataSeries::zero ()
{
  memory->do_zero (buffer, size);
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

void dsp::DataSeries::reshape (unsigned new_npol, unsigned new_ndim)
{
  unsigned new_total = new_npol * new_ndim;
  unsigned total = get_npol() * get_ndim();

  if (total < new_total)
    throw Error (InvalidParam, "dsp::DataSeries::reshape",
		 "current npol=%d*ndim=%d = %d != new npol=%d*ndim=%d = %d",
		 get_npol(), get_ndim(), total, new_npol, new_ndim, new_total);

  subsize *= get_npol();
  subsize /= new_npol;

  set_npol (new_npol);
  set_ndim (new_ndim);
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

  return get_data() + (ichan*get_npol() + ipol) * subsize;
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

  uint64_t npt = (get_ndat() * get_ndim() * get_nbit())/8;

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
  // Observation::swap_data( ts );
  unsigned char* tmp = buffer; buffer = ts.buffer; ts.buffer = tmp;
  uint64_t tmp2 = size; size = ts.size; ts.size = tmp2;
  uint64_t tmp3 = subsize; subsize = ts.subsize; ts.subsize = tmp3;

  if( subsize*get_npol()*get_nchan() > size )
    throw Error(InvalidState,"dsp::DataSeries::swap_data()",
		"BUG! subsize*get_npol()*get_nchan() > size ("UI64" * %d * %d > "UI64")\n",
		subsize,get_npol(),get_nchan(),size);

  return *this;
}

//! Returns the number of samples that have been seeked over
int64_t dsp::DataSeries::get_samps_offset() const {
  uint64_t bytes_offset = get_data() - buffer; 
  uint64_t samps_offset = (bytes_offset * 8)/(get_nbit()*get_ndim());
  //  fprintf(stderr,"\ndsp::DataSeries::get_samps_offset() got bytes_offset="UI64" nbit=%d ndim=%d so samp offset="UI64"\n",
  //  bytes_offset,get_nbit(),get_ndim(),samps_offset);
  return int64_t(samps_offset);
}

//! Returns the maximum ndat possible with current offset of data from base pointer
uint64_t dsp::DataSeries::maximum_ndat() const {
  uint64_t bytes_offset = get_data() - buffer; 
  uint64_t bits_avail = (subsize-bytes_offset) * 8;
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


//! Match the internal memory layout of another DataSeries
void dsp::DataSeries::internal_match (const DataSeries* other)
{
  if (size < other->size)
  {
    if (verbose)
      cerr << "dsp::DataSeries::internal_match Memory::free"
	" size=" << size << " buffer=" << (void*)buffer << endl;
    memory->do_free (buffer);
    memory_used -= size;

    if (verbose)
      cerr << "dsp::DataSeries::internal_match"
	" Memory::allocate (" << other->size << ")" << endl;

    buffer = (unsigned char*) memory->do_allocate (other->size);
    if (!buffer)
      throw Error (InvalidState,"dsp::DataSeries::internal_match",
		  "could not allocate "UI64" bytes", other->size);

    size = other->size;
    memory_used += size;
  }

  subsize = other->subsize;
  copy_dimensions( other );
}

