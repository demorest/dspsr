/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TransferCUDA.h"

#include "Error.h"

//! Default constructor- always inplace
dsp::TransferCUDA::TransferCUDA()
  : Transformation<TimeSeries,TimeSeries> ("CUDA::Transfer",outofplace,true)
{
  kind = cudaMemcpyHostToDevice;
}

//! Do stuff
void dsp::TransferCUDA::transformation ()
{
  const unsigned npol = input->get_npol();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim = input->get_ndim();
  const uint64_t ndat = input->get_ndat();

  const uint64_t nbyte = ndat * ndim * sizeof(float);

  prepare ();

  unsigned ichan = 0;
  unsigned ipol = 0;

  const float* from = input->get_datptr( ichan, ipol );
  float* to = output->get_datptr( ichan, ipol );

  // check for contiguity

  if (nchan > 1)
    ichan = 1;
  else if (npol > 1)
    ipol = 1;

  if (ichan || ipol)
  {
    // check the pointers of the next blocks
    const float* from2 = input->get_datptr( ichan, ipol );
    float* to2 = input->get_datptr( ichan, ipol );

    if ( (from2 - from == ndat * ndim)
	 && (to2 - to == ndat * ndim) )
    {
      cerr << "dsp::TransferCUDA::transformation contiguous blocks" << endl;
      nbyte *= npol * nchan;
      cudaMemcpy (to, from, nbyte, kind);
      return;
    }
  }

  for (unsigned ipol=0; ipol < npol; ipol++)
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      void* to = output->get_datptr( ichan, ipol );
      cudaMemcpy (to, from, nbyte, kind);
    }
}

void dsp::TransferCUDA::prepare ()
{
  output->copy_configuration( input );
  output->set_input_sample( input->get_input_sample() );
  output->resize( input->get_ndat () );
}

