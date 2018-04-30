/***************************************************************************
 *
 *   Copyright (C) 2017 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/VDIFnByteUnpacker.h"
#include "Error.h"

#include <assert.h>

using namespace std;

//! Constructor
dsp::VDIFnByteUnpacker::VDIFnByteUnpacker (const char* name)
  : Unpacker (name)
{
}

bool dsp::VDIFnByteUnpacker::matches (const Observation* observation)
{
#ifdef _DEBUG
  cerr << "dsp::VDIFnByteUnpacker::matches machine=" << observation->get_machine()
       << " nbit=" << observation->get_nbit()
       << " ndim=" << observation->get_ndim() << endl;
#endif

  // supports any VDIF file with nbit = k * 8
  return observation->get_machine() == "VDIF" 
    && (observation->get_nbit() == 16 || observation->get_nbit() == 32);

}

template<typename T>
void convert_to_float ( unsigned ndat, unsigned ndim, unsigned nspan,
			float* into, T* from)
{
  for (unsigned idat = 0; idat < ndat; idat++)
  {
    for (unsigned idim = 0; idim < ndim; idim++)
      into[idat*ndim+idim] = (float) from[idim]; // deal with Endian here

    from += nspan;
  }
}

void dsp::VDIFnByteUnpacker::unpack ()
{
  const uint64_t ndat  = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();
  const unsigned nbit = input->get_nbit();
  const unsigned nbyte = nbit / 8;
  
  const unsigned char* base = input->get_rawptr();

  for (unsigned ichan=0; ichan<nchan; ichan++) 
  {
    for (unsigned ipol=0; ipol<npol; ipol++) 
    {
      const unsigned char* from = base + (ichan*npol + ipol) * ndim * nbyte;
      float* into = output->get_datptr (ichan, ipol);
	
      switch (nbit)
	{
	case 16:
	  convert_to_float (ndat, ndim, nchan*npol*ndim, into,
			    reinterpret_cast<const uint16_t*> (from));
	  break;
	  
	case 32:
	  convert_to_float (ndat, ndim, nchan*npol*ndim, into,
			    reinterpret_cast<const uint32_t*> (from));

	case 64:
	  convert_to_float (ndat, ndim, nchan*npol*ndim, into,
			    reinterpret_cast<const uint64_t*> (from));	  
	}
    }
  }
}
