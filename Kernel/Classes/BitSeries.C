/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BitSeries.h"
#include "Error.h"
#include "tostring.h"

#include <string.h>

using namespace std;

//! Null constructor
dsp::BitSeries::BitSeries ()
{
  data = 0;
  data_size = 0;
  input_sample = -1;
  input = 0;

  request_offset = 0;
  request_ndat = 0;

  memory = Memory::get_manager ();
}

//! Destructor
dsp::BitSeries::~BitSeries ()
{
  if (data) memory->do_free(data); data = 0;
  data_size = 0;
}

void dsp::BitSeries::set_memory (Memory* m)
{
  memory = m;
}

//! Allocate the space required to store nsamples time samples.
/*!
  \pre The dimensions of each time sample (nchan, npol, ndim, nbit) should
  have been set prior to calling this method (see Observation::set_sample).
  \post There is no guarantee that the data already contained in BitSeries
  will be preserved after a call to resize.
*/
void dsp::BitSeries::resize (int64_t nsamples)
{
  int64_t require = get_nbytes(nsamples);

  if (require < 0)
    throw Error (InvalidParam, "dsp::BitSeries::resize",
		 "invalid size="I64, require);

  if (!require || require > data_size) {

    if (verbose)
      cerr << "dsp::BitSeries::resize current size = " << data_size << " bytes"
          " -- required size = " << require << " bytes" << endl;
 
    if (data) memory->do_free( data ); data = 0;
    data_size = 0;
    //! data has been deleted. input sample is no longer valid
    input_sample = -1;
    input = 0;
  }

  if( verbose )
    fprintf(stderr,"dsp::BitSeries::resize() setting request_ndat to "I64"\n",
	  nsamples);
  
  set_ndat( nsamples );
  request_ndat = nsamples;
  request_offset = 0;

  if (!require)
    return;

  if (data_size == 0) {
    data = (unsigned char*) memory->do_allocate (require);
    data_size = require;
  }

}

dsp::BitSeries& 
dsp::BitSeries::operator = (const BitSeries& bitseries)
{
  if (this == &bitseries)
    return *this;

  Observation::operator = (bitseries);
  resize (bitseries.get_ndat());

  const unsigned char* from = bitseries.get_rawptr();
  unsigned char* to = get_rawptr();

  memcpy(to,from,size_t(bitseries.get_nbytes()));

  return *this;
}


//! Return pointer to the specified data block
unsigned char* dsp::BitSeries::get_datptr(uint64_t sample)
{
  return data + get_nbytes(sample);
}

//! Return pointer to the specified data block
const unsigned char* dsp::BitSeries::get_datptr(uint64_t sample) const
{
  return data + get_nbytes(sample);
}

int64_t dsp::BitSeries::get_input_sample (Input* test_input) const
{
  if (test_input && test_input != input)
    return -1;
  
  return input_sample; 
}

void dsp::BitSeries::copy_data (const dsp::BitSeries* copy, 
				uint64_t idat_start, uint64_t copy_ndat) try
{
  if (verbose)
    cerr << "dsp::BitSeries::copy_data to ndat=" << get_ndat()
	 << " from ndat=" << copy->get_ndat() 
	 << "\n  idat_start=" << idat_start 
	 << " copy_ndat=" << copy_ndat << endl;

  if (copy_ndat > get_ndat())
    throw Error (InvalidParam, "dsp::BitSeries::copy_data",
		 "copy ndat="UI64" > this ndat="UI64, copy_ndat, get_ndat());

  if (copy_ndat)
  {
    uint64_t bytes = get_nbytes (copy_ndat);
    uint64_t offset = get_nbytes (idat_start);

    if (verbose)
      cerr << "dsp::BitSeries::copy_data"
	" bytes=" << bytes << " offset=" << offset << endl;

    unsigned char *into = get_rawptr();
    const unsigned char *from = copy->get_rawptr() + offset;

    memcpy (into, from, size_t(bytes));
  }

  input_sample = copy->input_sample + idat_start;
  input = copy->input;
}
catch (Error& error)
{
  throw error += "dsp::BitSeries::copy_data";
}

void dsp::BitSeries::append (const dsp::BitSeries* little)
{
  if( !get_ndat() ){
    if( data_size < little->data_size )
      throw Error(InvalidRange,"dsp::BitSeries::append()",
		  "BitSeries does not have required capacity to be appended to (size=" + tostring(data_size) + ")");

    Observation::operator=(*little);
    set_ndat(0);
  }    

  else{
    if( !combinable(*little) )
      throw Error(InvalidState,"dsp::BitSeries::append()",
		  "BitSerieses not combinable");
    
    if( get_nsamples(data_size) < get_ndat()+little->get_ndat() )
      throw Error(InvalidRange,"dsp::BitSeries::append()",
		  "BitSeries does not have required capacity to be appended to");
  }    

  const unsigned char* from = little->get_datptr(0);
  unsigned char* to = get_datptr(get_ndat());

  memcpy(to,from,size_t(get_nbytes()));

  set_ndat( get_ndat() + little->get_ndat() );
}

