/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/Dump.h"

#if HAVE_CUDA
#include "dsp/TransferCUDA.h"
#include "dsp/MemoryCUDA.h"
#endif

#include <fstream>
using namespace std;

dsp::Dump::Dump (const char* name) : Sink<TimeSeries> (name)
{
  binary = false;
}

dsp::Dump::~Dump ()
{
  if (output)
    cerr << "dsp::Dump::~Dump fptr=" << (FILE*) output << endl;
}

//! Set the ostream to which data will be dumped
void dsp::Dump::set_output (FILE* fptr)
{
  cerr << "dsp::Dump::set_output fptr=" << (void*)fptr << endl;
  output = fptr;
}

//! Set the flag to output binary data
void dsp::Dump::set_output_binary (bool flag)
{
  binary = flag;
}

//! Adds to the totals
void dsp::Dump::calculation ()
{
  Reference::To<const TimeSeries> use = input;

#if HAVE_CUDA
  if ( dynamic_cast<const CUDA::DeviceMemory*>( input->get_memory() ) )
  {
    cerr << "dsp::Dump::calculate retrieving from GPU" << endl;
    TransferCUDA transfer;
    transfer.set_kind( cudaMemcpyDeviceToHost );
    transfer.set_input( input );
	
    Reference::To<TimeSeries> host = new TimeSeries;
    transfer.set_output( host );
    transfer.operate ();

    use = host;
  }
#endif

  const unsigned nchan = use->get_nchan();
  const unsigned npol = use->get_npol();
  const uint64_t ndat = use->get_ndat();
  const unsigned ndim = use->get_ndim();

  const int64_t istart = use->get_input_sample();

  if (verbose)
    cerr << "dsp::Dump::calculate nchan=" << nchan << " npol=" << npol 
         << " ndat=" << ndat << " ndim=" << ndim << endl; 

  for (unsigned ichan = 0; ichan < nchan; ichan ++)
  {
    for (unsigned ipol = 0; ipol < npol; ipol++)
    {
      const float* data = use->get_datptr (ichan, ipol);

      if (output)
      {
#if 0
	fwrite (&ichan,  sizeof(unsigned), 1, output);
	fwrite (&ipol,   sizeof(unsigned), 1, output);
	fwrite (&istart, sizeof(int64_t),  1, output);
	fwrite (&ndat,   sizeof(uint64_t), 1, output);
	fwrite (data,    sizeof(float), ndat, output);
#else
  for (unsigned i=0; i < 10; i++)
  {
    fprintf (output, "%u %u %u %f", ichan, ipol, istart+i, data[i*ndim]);
    for (unsigned j=1; j<ndim; j++)
      fprintf (output, " %f", data[i*ndim+j]);
    fprintf (output, "\n");
  }
#endif
	continue;
      }

#if 0
      float min = std::numeric_limits<float>::max();
      float max = -min;
      double tot = 0.0;
      double totsq = 0.0;

      for (uint64_t idat = 0; idat < ndat*ndim; idat++)
      {
	if (data[idat] < min)
	  min = data[idat];
	if (data[idat] > max)
	  max = data[idat];

	tot += data[idat];
	totsq += data[idat] * data[idat];
      }

      double mean = tot / (ndat*ndim);
      double var = totsq / (ndat*ndim) - mean * mean;

      cerr << "ichan=" << ichan << " ipol=" << ipol
	   << " min=" << min << " max=" << max
	   << " mean=" << tot/(ndat*ndim) << " rms=" << sqrt(var) << endl;
#endif

      for (uint64_t idat = 0; idat < ndat*ndim; idat++)
	if (!finite(data[idat]))
	  cerr << "NaN/Inf ichan=" << ichan << " ipol=" << ipol
	       << " ifloat=" << idat << endl;

    }
  }
}

