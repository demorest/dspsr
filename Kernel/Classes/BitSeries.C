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
}

//! Destructor
dsp::BitSeries::~BitSeries ()
{
  if (data) delete [] data; data = 0;
  data_size = 0;
}

//! Allocate the space required to store nsamples time samples.
/*!
  \pre The dimensions of each time sample (nchan, npol, ndim, nbit) should
  have been set prior to calling this method (see Observation::set_sample).
  \post There is no guarantee that the data already contained in BitSeries
  will be preserved after a call to resize.
*/
void dsp::BitSeries::resize (int64 nsamples)
{
  int64 require = get_nbytes(nsamples);

  if (require < 0)
    throw Error (InvalidParam, "dsp::BitSeries::resize",
		 "invalid size="I64, require);

  if (!require || require > data_size) {

    if (verbose)
      cerr << "dsp::BitSeries::resize current size = " << data_size << " bytes"
          " -- required size = " << require << " bytes" << endl;
 
    if (data) delete [] data; data = 0;
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
    data = new unsigned char [require];
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
unsigned char* dsp::BitSeries::get_datptr(uint64 sample)
{
  return data + get_nbytes(sample);
}

//! Return pointer to the specified data block
const unsigned char* dsp::BitSeries::get_datptr(uint64 sample) const
{
  return data + get_nbytes(sample);
}

int64 dsp::BitSeries::get_input_sample (Input* test_input) const
{
  if (test_input && test_input != input)
    return -1;
  
  return input_sample; 
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

//! Delete the current data buffer and attach to this one
//! This is dangerous as it ASSUMES new data buffer has been pre-allocated and is big enough.  Beware of segmentation faults when using this routine.
//! Also do not try to delete the old memory once you have called this- the BitSeries::data member now owns it.
void dsp::BitSeries::attach(auto_ptr<unsigned char> _data){
  if( !_data.get() )
    throw Error(InvalidState,"dsp::BitSeries::attach()",
		"NULL auto_ptr has been passed in- you haven't properly allocated it using 'new' before passing it into this method");

  if (data) delete [] data; data = 0;
  data = _data.release();
}

//! Call this when you want the array to still be owned by it's owner
void dsp::BitSeries::attach(unsigned char* _data){
  if( !_data )
    throw Error(InvalidState,"dsp::BitSeries::attach()",
		"NULL ptr has been passed in- you haven't properly allocated it using 'new' before passing it into this method");

  if (data) delete [] data; data = 0;
  data = _data;
}

void dsp::BitSeries::share(unsigned char*& _buffer,uint64& _size) const {
  _size = data_size;
  _buffer = data;
}

//! Release control of the data buffer- resizes to zero
unsigned char* dsp::BitSeries::release(uint64& _size){
  if( !data )
    return 0;

  unsigned char* ret = data;
  _size = data_size;

  data = 0;
  data_size = 0;
  input_sample = -1;
  input = 0;
  set_ndat( 0 );
  request_ndat = 0;
  request_offset = 0;

  return ret;
}

