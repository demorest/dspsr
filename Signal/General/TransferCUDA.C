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
  const unsigned ndim = input->get-ndim();
  const uint64_t ndat = input->get_ndat();

  const uint64_t nbyte = ndat * ndim * sizeof(float);

  output->copy_configuration(input);
  
  for (unsigned ipol=0; ipol < npol; ipol++)
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      void* to = output->get_datptr( ichan, ipol );
      const void* from = input->get_datptr( ichan, ipol );
      cudaMemcpy (to, from, nbyte, kind);
    }
}
