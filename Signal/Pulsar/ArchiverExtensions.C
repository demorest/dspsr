/***************************************************************************
 *
 *   Copyright (C) 2003-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Archiver.h"
#include "dsp/Response.h"
#include "dsp/PhaseSeries.h"

#include "dsp/SignalPath.h"
#include "dsp/Input.h"
#include "dsp/IOManager.h"
#include "dsp/ExcisionUnpacker.h"
#include "dsp/Filterbank.h"
#include "dsp/Convolution.h"
#include "dsp/TScrunch.h"

#include "Pulsar/dspReduction.h"
#include "Pulsar/TwoBitStats.h"
#include "Pulsar/Passband.h"

#include "Error.h"

using namespace std;

void dsp::Archiver::set (Pulsar::dspReduction* dspR) try
{
  const char* method = "dsp::Archiver::set Pulsar::dspReduction";

  if (!profiles)
    throw Error (InvalidState, method, "Profile data not provided");

  if (verbose > 2)
    cerr << method << " start" << endl;

  dspR->set_software( archive_software );
  dspR->set_name( profiles->get_machine() );
  dspR->set_scale( profiles->get_scale() );

  if (!profiles->has_extensions())
  {
    if (verbose > 2)
      cerr << method << " no Extensions" << endl;
    return;
  }

  const SignalPath* path = profiles->get_extensions()->get<SignalPath>();

  if (!path)
  {
    if (verbose > 2)
      cerr << method << " no SignalPath" << endl;
    return;
  }

  const SignalPath::List* list = path->get_list();

  if (!list || !list->size())
  {
    if (verbose > 2)
      cerr << method << " empty SignalPath" << endl;
    return;
  }

  for (unsigned i = 0; i < list->size(); i++)
  {
    Operation* operation = (*list)[i];

    // ////////////////////////////////////////////////////////////////////
    //
    // Input class parameters
    //
    Input* input = dynamic_cast<Input*>( operation );

    if (input)
    {
      dspR->set_total_samples ( input->get_total_samples() );
      dspR->set_block_size ( input->get_block_size() );
      dspR->set_overlap ( input->get_overlap() );
    }

    // ////////////////////////////////////////////////////////////////////
    //
    // TwoBitCorrection class parameters
    //
    const ExcisionUnpacker* excision = 0;

    // ////////////////////////////////////////////////////////////////////
    //
    // IOManager class may contain an ExcisionUnpacker
    //
    IOManager* manager = dynamic_cast<IOManager*>( operation );
    if (manager)
      operation = manager->get_unpacker();

    excision = dynamic_cast<const ExcisionUnpacker*> ( operation );

    // save it for the TwoBitStats Extension
    if (excision)
      excision_unpacker = excision;

    // ////////////////////////////////////////////////////////////////////
    //
    // Filterbank class parameters
    //
    Filterbank* filterbank = dynamic_cast<Filterbank*>( operation );

    if (filterbank)
    {
      dspR->set_nchan ( filterbank->get_nchan() );
      dspR->set_freq_res ( filterbank->get_freq_res() );
      dspR->set_time_res ( filterbank->get_time_res() );
    }

    // ////////////////////////////////////////////////////////////////////
    //
    // Convolution class parameters
    //

    Convolution* convolution = dynamic_cast<Convolution*>(operation);

    if (convolution)
    {
      if ( convolution->has_response() )
      {
        if (verbose > 2) cerr << method << " Convolution with Response" << endl;

        const Response* response = convolution->get_response ();

        unsigned nsamp_fft = response->get_ndat();
        unsigned nsamp_overlap_pos = response->get_impulse_pos ();
        unsigned nsamp_overlap_neg = response->get_impulse_neg ();

        const TimeSeries* input = convolution->get_input ();

        if (input->get_state() == Signal::Nyquist)
	{
	  nsamp_fft *= 2;
	  nsamp_overlap_pos *= 2;
	  nsamp_overlap_neg *= 2;
        }

        dspR->set_nsamp_fft ( nsamp_fft );
        dspR->set_nsamp_overlap_pos ( nsamp_overlap_pos );
        dspR->set_nsamp_overlap_neg ( nsamp_overlap_neg );
      }

      // save it for the Passband Extension
      if ( convolution->has_passband() )
	passband = convolution->get_passband();
    }

    // ////////////////////////////////////////////////////////////////////
    //
    // Tscrunch class parameters
    //
    TScrunch* tscrunch = dynamic_cast<TScrunch*>( operation );

    if (tscrunch)
      dspR->set_ScrunchFactor ( tscrunch->get_factor() );
  }

}
catch (Error& error)
{
  throw error += "dsp::Archiver::set Pulsar::dspReduction";
}


void dsp::Archiver::set (Pulsar::TwoBitStats* tbc) try
{
  if (verbose > 2)
    cerr << "dsp::Archiver::set Pulsar::TwoBitStats Extension" << endl;

  if (!excision_unpacker)
  {
    if (verbose > 2)
      cerr << "dsp::Archiver::set Pulsar::TwoBitStats no ExcisionUnpacker"
	   << endl;
    return;
  }

  unsigned ndig = excision_unpacker->get_ndig ();
  unsigned ndat_per_weight = excision_unpacker->get_ndat_per_weight ();

  tbc->resize (ndat_per_weight, ndig);

  tbc->set_threshold ( excision_unpacker->get_threshold() );
  tbc->set_cutoff_sigma ( excision_unpacker->get_cutoff_sigma() );

  // temporary space
  vector<float> histogram;

  for (unsigned idig=0; idig<ndig; idig++)
  {
    excision_unpacker->get_histogram (histogram, idig);
    tbc->set_histogram (histogram, idig);
  }
}
catch (Error& error)
{
  throw error += "dsp::Archiver::set Pulsar::TwoBitStats";
}


void dsp::Archiver::set (Pulsar::Passband* pband) try
{
  if (verbose > 2)
    cerr << "dsp::Archiver::set Pulsar::Passband Extension" << endl;

  if (!passband)
  {
    if (verbose > 2)
      cerr << "dsp::Archiver::set Pulsar::Passband no passband" << endl;
    return;
  }

  // terminology differs between dsp::Shape and the Pulsar::Passband Extension
  unsigned npol = passband->get_npol ();
  unsigned nband = passband->get_nchan ();
  unsigned nchan = passband->get_ndat ();

  if (npol==0 || nband==0 || nchan==0)
  {
    if (verbose > 2)
      cerr << "dsp::Archiver::set Pulsar::Passband empty passband" << endl;
    return;
  }

  dsp::Response copy (*passband);
  copy.naturalize ();
  
  if (verbose > 2)
    cerr << "dsp::Archiver::set Pulsar::Passband Extension copy\n\t"
            " npol=" << npol << " nband=" << nband <<
            " nchan/band=" << nchan << endl;
  
  pband->resize (nchan, npol, nband);
  
  if (passband->get_ndim() != 1)
    throw Error (InvalidState, "dsp::Archiver::set_passband",
		 "Passband Response ndim != 1");
  
  for (unsigned ipol=0; ipol<npol; ipol++)
    for (unsigned iband=0; iband<nband; iband++)
      pband->set_passband (copy.get_datptr (iband, ipol), ipol, iband);
}
catch (Error& error)
{
  throw error += "dsp::Archiver::set Pulsar::Passband";
}


