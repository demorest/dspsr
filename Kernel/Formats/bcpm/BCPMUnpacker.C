/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BCPMUnpacker.h"
#include "dsp/BCPMFile.h"

#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"

#include "machine_endian.h"
#include "Error.h"

#include <string.h>

using namespace std;

//! Null constructor
dsp::BCPMUnpacker::BCPMUnpacker (const char* _name) : Unpacker (_name){ }

bool dsp::BCPMUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "BCPM";
}

//! Does the work
void dsp::BCPMUnpacker::unpack ()
{
  if (!file)
    file = get_Input<BCPMFile>();

  if( get_input()->get_nbit() != 4 )
    throw Error(InvalidState,"dsp::BCPMUnpacker::unpack ()",
		"Input nbit=%d.  Only a 4-bit unpacker is written",
		get_input()->get_nbit());

  get_output()->Observation::operator=( *get_input() );
  get_output()->set_nbit( 32 );
  get_output()->resize( get_input()->get_ndat() );

  const vector<int>& chtab = file->chtab;

  const unsigned nchan = get_input()->get_nchan();
  vector<float> tempblock(nchan);
  const unsigned char* raw = get_input()->get_rawptr();

  if( get_input()->get_nbytes() != get_input()->get_ndat() * nchan / 2 )
    throw Error(InvalidState,"dsp::BCPMUnpacker::unpack()",
		"Loop assumption incorrect.  Bug!");

  static float512 lookup = get_lookup();

  vector<float*> datptrs( nchan );
  for( unsigned i=0; i<nchan; i++)
    datptrs[i] = get_output()->get_datptr(i,0);

  vector<unsigned char> temp_buffer(nchan/2);
  unsigned char* in = &temp_buffer[0];

  for (unsigned s=0; s<unsigned(get_input()->get_ndat()); s++)
  {
    memcpy(in, raw, nchan/2);

    for (unsigned l=0; l<nchan/2; l++) { FromBigEndian(in[l]) ; }

    unsigned j = 0;
    for( unsigned i=0; i<nchan/2; i++){
      tempblock[j] = lookup.data[in[i]]; j++;
      tempblock[j] = lookup.data[256+in[i]]; j++;
    }

    for( unsigned k=0; k<nchan; k++)
      datptrs[k][s] = tempblock[chtab[k]];

    raw += nchan/2;
  }
}

//! Generates the lookup table
dsp::BCPMUnpacker::float512 dsp::BCPMUnpacker::get_lookup()
{
  float512 lookup;
  
  float lower = 0.0;
  float step = 1.0;
  
  for( unsigned char i=0; i<255; ++i){
    lookup.data[i] =     lower + step*float((i >> 4) & 15);
    lookup.data[256+i] = lower + step*float( i       & 15);
  }
  lookup.data[255] =     lower + 15.*step;
  lookup.data[256+255] = lower + 15.*step;
  
  return lookup;
}
