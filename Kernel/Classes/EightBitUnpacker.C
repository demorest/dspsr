/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/EightBitUnpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

// #define _DEBUG 1

using namespace std;

//! Null constructor
dsp::EightBitUnpacker::EightBitUnpacker (const char* _name)
  : BitUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::EightBitUnpacker ctor" << endl;
}

void dsp::EightBitUnpacker::unpack (uint64_t ndat, 
				    const unsigned char* from,
				    const unsigned nskip,
				    float* into, 
                                    const unsigned fskip,
				    unsigned long* hist)
{
  const float* lookup = table->get_values ();

  if (verbose)
    cerr << "dsp::EightBitUnpacker::unpack ndat=" << ndat << endl;

  for (uint64_t idat = 0; idat < ndat; idat++)
  {
    hist[ *from ] ++;
    *into = lookup[ *from ];

#ifdef _DEBUG
    cerr << idat << " " << int(*from) << "=" << *into << endl;
#endif

    from += nskip;
    into += fskip;
  }
}

