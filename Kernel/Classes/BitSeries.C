#include <string.h>

#include "genutil.h"
#include "string_utils.h"

#include "dsp/BitSeries.h"
#include "Error.h"

//! Null constructor
dsp::BitSeries::BitSeries ()
{
  data = 0;
  size = 0;
  input_sample = -1;
}

//! Destructor
dsp::BitSeries::~BitSeries ()
{
  if (data) delete [] data; data = 0;
  size = 0;
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
  int64 require = nbytes (nsamples);

  if (require < 0)
    throw_str ("BitSeries::resize invalid size="I64, require);

  if (!require || require > size) {
    if (data) delete [] data; data = 0;
    size = 0;
    //! data has been deleted. input sample is no longer valid
    input_sample = -1;
  }

  ndat = nsamples;

  if (!require)
    return;

  if (size == 0) {
    data = new unsigned char [require];
    size = require;
  }

}

dsp::BitSeries& 
dsp::BitSeries::operator = (const BitSeries& basicseries)
{
  if (this == &basicseries)
    return *this;

  Observation::operator = (basicseries);
  resize (basicseries.ndat);

  const unsigned char* from = basicseries.get_rawptr();
  unsigned char* to = get_rawptr();

  memcpy(to,from,basicseries.nbytes());

  return *this;
}


//! Return pointer to the specified data block
unsigned char* dsp::BitSeries::get_datptr(uint64 sample)
{
  return data + nbytes(sample);
}

//! Return pointer to the specified data block
const unsigned char* dsp::BitSeries::get_datptr(uint64 sample) const
{
  return data + nbytes(sample);
}

void dsp::BitSeries::append (const dsp::BitSeries* little)
{
  if( !ndat ){
    if( size < little->size )
      throw Error(InvalidRange,"dsp::BitSeries::append()",
		  string("BitSeries does not have required capacity to be appended to (size=")
		  + make_string(size) + string(")"));

    Observation::operator=(*little);
    ndat = 0;
  }    

  else{
    if( !combinable(*little) )
      throw Error(InvalidState,"dsp::BitSeries::append()",
		  "BitSerieses not combinable");
    
    if( size/nbytes(1) < ndat+little->ndat )
      throw Error(InvalidRange,"dsp::BitSeries::append()",
		  "BitSeries does not have required capacity to be appended to");
  }    

  const unsigned char* from = little->get_datptr(0);
  unsigned char* to = get_datptr(ndat);

  memcpy(to,from,nbytes());

  ndat += little->ndat;
  
}
