//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/MultiplexUnpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

using namespace std;

dsp::MultiplexUnpacker::MultiplexUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::MultiplexUnpacker ctor" << endl;
  set_nstate (256);
  table = new BitTable (8, BitTable::TwosComplement);
}

dsp::MultiplexUnpacker::~MultiplexUnpacker ()
{
}

double dsp::MultiplexUnpacker::get_optimal_variance ()
{
  if (!table)
    throw Error (InvalidState, "dsp::MultiplexUnpacker::get_optimal_variance", 
		 "BitTable not set");

return table->get_optimal_variance();
}

void dsp::MultiplexUnpacker::set_table(BitTable* _table)
{
  if (verbose)
    cerr << "dsp::MultiplexUnpacker::set_table" << endl;

  table = _table;
}

const dsp::BitTable* dsp::MultiplexUnpacker::get_table() const
{
  return table;
}

bool dsp::MultiplexUnpacker::matches (const Observation* observation)
{
  //  return observation->get_machine()== "MULTIPLEX"
  //  && observation->get_nbit() == 8;

  return observation->get_machine()== "CASPSR"
    && observation->get_nbit() == 8;

  
}

void dsp::MultiplexUnpacker::unpack(uint64_t ndat,
				    const unsigned char* from,
				    const unsigned nskip,
				    float* into,
				    const unsigned fskip,
				    unsigned long* hist)
{
  const float* lookup = table->get_values ();
  int counter_four = 0;

  if (verbose)
    cerr << "dsp::MultiplexUnpacker::unpack ndat=" << ndat << endl;

  for (uint64_t idat = 0; idat < ndat; idat++)
  {
    hist[ *from ] ++;
    *into = lookup[ *from ];
    
#ifdef _DEBUG
    cerr << idat << " " << int(*from) << "=" << *into << endl;
#endif
    counter_four++;
    if (counter_four == 4)
      {
	counter_four = 0;
	from += 5; //(nskip+4);
      }
    else
      {
	from ++; //=nskip;
      }
    into += fskip;
  }
}

void dsp::MultiplexUnpacker::unpack ()
{
  const uint64_t   ndat  = input->get_ndat();
  
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();
  
  const unsigned nskip = npol * nchan * ndim;
  const unsigned fskip = ndim;
  
  unsigned offset = 0;
  
  for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      for (unsigned ipol=0; ipol<npol; ipol++)
	{
	  if (ipol==1)
	    offset = 4;
	  for (unsigned idim=0; idim<ndim; idim++)
	    {
	      const unsigned char* from = input->get_rawptr() + offset;
	      float* into = output->get_datptr (ichan, ipol) + idim;
	      unsigned long* hist = get_histogram (ipol);
	      
#ifdef _DEBUG
	      cerr << "c=" << ichan << " p=" << ipol << " d=" << idim << endl;
#endif
	      
	      unpack (ndat, from, nskip, into, fskip, hist);
	      offset ++;
	    }
	}
    }
}
