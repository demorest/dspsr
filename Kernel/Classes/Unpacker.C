#include <stdio.h>
#include <stdlib.h>

#include "dsp/Unpacker.h"
#include "Error.h"

//! Initialize and resize the output before calling unpack
void dsp::Unpacker::transformation ()
{
  if (verbose)
    cerr << "dsp::Unpacker::transformation" << endl;;

  remove_subheaders();

  // set the Observation information
  output->Observation::operator=(*input);

  // resize the output 
  output->resize (input->get_ndat());

  // unpack the data
  unpack ();

  // The following lines deal with time sample resolution of the data source
  output->seek (input->get_request_offset());

  output->set_ndat (input->get_request_ndat());

  if (verbose)
    cerr << "dsp::Unpacker::transformation exit" << endl;;
}

//! Specialize the unpacker to the Observation
void dsp::Unpacker::match (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::Unpacker::match" << endl;
}

//! Removes subheaders
void dsp::Unpacker::remove_subheaders(){
  uint64 data_bytes = get_input()->get_data_bytes();
  uint64 subheader_bytes = get_input()->get_subheader_bytes();
  uint64 bytes_since_subheader = get_input()->get_bytes_since_subheader();
  uint64 input_nbytes = get_input()->get_nbytes();

  if( subheader_bytes==0 )
    return;

  uint64 block_size = data_bytes + subheader_bytes;

  if( bytes_since_subheader > block_size )
    throw Error(InvalidState,"dsp::Unpacker::remove_subheaders()",
		"bytes_since_subheader > block_size ("UI64" > "UI64") which should be impossible",
		bytes_since_subheader , block_size);

  if( input_nbytes <= subheader_bytes )
    throw Error(InvalidState,"dsp::Unpacker::remove_subheaders()",
		"input_nbytes <= subheader_bytes");

  uint64 subheader_byte_number = block_size - bytes_since_subheader;

  bool in_header = false;
  if( bytes_since_subheader < subheader_bytes )
    in_header = true;

  if( !in_header && subheader_byte_number > input_nbytes )
    return;

  Reference::To<BitSeries> good(new BitSeries);
  good->operator=( *get_input() );
  good->set_ndat( 0 );

  uint64 input_bytes_done = 0;
  uint64 output_bytes_done = 0;

  unsigned char* to = good->get_rawptr();
  unsigned char* from = (unsigned char*)input->get_rawptr();

  while( input_bytes_done < input_nbytes ){
    if( in_header ){
      input_bytes_done = subheader_bytes - bytes_since_subheader;
      in_header = false;
      continue;
    }
      
    uint64 bytes_before_subheader = subheader_byte_number - input_bytes_done;

    uint64 bytes_to_memcpy = bytes_before_subheader;
    if( input_bytes_done + bytes_to_memcpy > input_nbytes )
      bytes_to_memcpy = input_nbytes - input_bytes_done;

    memcpy(to+output_bytes_done,from+input_bytes_done,bytes_to_memcpy);

    input_bytes_done += bytes_to_memcpy + subheader_bytes;
    output_bytes_done += bytes_to_memcpy;
    subheader_byte_number += block_size;
  }

  good->set_ndat( good->get_nsamples(output_bytes_done) );

  (const_cast<BitSeries*>(input.get()))->operator=( *good );
}
