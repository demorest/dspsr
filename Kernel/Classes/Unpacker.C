#include "dsp/Unpacker.h"
#include "Error.h"

//! Initialize and resize the output before calling unpack
void dsp::Unpacker::transformation ()
{
  if (verbose)
    cerr << "dsp::Unpacker::transformation" << endl;;

  // set the Observation information
  output->Observation::operator=(*input);

  //fprintf(stderr,"hi\n");

  // resize the output 
  output->resize (input->get_ndat());

  //fprintf(stderr,"hi1\n");

  // unpack the data
  unpack ();

  //fprintf(stderr,"hi2\n");

  // The following lines deal with time sample resolution of the data source
  // HSK 8 Feb 2003 commented this line out too as it looks bad news
  //output->seek (input->get_request_offset());

  //fprintf(stderr,"In dsp::Unpacker::transformation() [4] with output->ndat="UI64"\n",
  //  output->get_ndat());

  // HSK 8 Feb 2003 this line is a bad, bad, bad idea.  Prefixing the outcome of a load??!?!?
  //output->set_ndat (input->get_request_ndat());

  if (verbose)
    cerr << "dsp::Unpacker::transformation exit" << endl;;
}

//! Specialize the unpacker to the Observation
void dsp::Unpacker::match (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::Unpacker::match" << endl;
}

