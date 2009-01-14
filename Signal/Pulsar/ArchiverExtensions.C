/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/Archiver.h"
#include "dsp/Response.h"
#include "dsp/PhaseSeries.h"

#include "dsp/Input.h"
#include "dsp/IOManager.h"
#include "dsp/TwoBitCorrection.h"
#include "dsp/Filterbank.h"
#include "dsp/Convolution.h"
#include "dsp/TScrunch.h"

#include "Pulsar/dspReduction.h"
#include "Pulsar/TwoBitStats.h"
#include "Pulsar/Passband.h"

#include "Error.h"

using namespace std;

void dsp::Archiver::set (Pulsar::dspReduction* dspR) try {

  if (verbose)
    cerr << "dsp::Archiver::set Pulsar::dspReduction Extension" << endl;

  dspR->set_software( archive_software );

  if (!operations.size()) {
    if (verbose)
      cerr << "dsp::Archiver::set Pulsar::dspReduction no operations" << endl;
    return;
  }

  if (!profiles)
    throw Error (InvalidState, "dsp::Archiver::set dspReduction Extension",
		 "Profile data not provided");

  for (unsigned i = 0; i < operations.size(); i++) {

    // ////////////////////////////////////////////////////////////////////
    //
    // Input class parameters
    //
    Input* input = dynamic_cast<Input*>( operations[i].get() );

    if (input) {

      dspR->set_total_samples ( input->get_total_samples() );
      dspR->set_block_size ( input->get_block_size() );
      dspR->set_overlap ( input->get_overlap() );

    }


    // ////////////////////////////////////////////////////////////////////
    //
    // TwoBitCorrection class parameters
    //
    const TwoBitCorrection* tbc;

    // ////////////////////////////////////////////////////////////////////
    //
    // IOManager class may contain a TwoBitCorrection Unpacker
    //
    IOManager* manager = dynamic_cast<IOManager*>( operations[i].get() );
    if (manager)
      tbc = dynamic_cast<const TwoBitCorrection*> ( manager->get_unpacker() );
    else
      tbc = dynamic_cast<const TwoBitCorrection*>( operations[i].get() );

    // save it for the TwoBitStats Extension
    if (tbc)
      twobit = tbc;


    // ////////////////////////////////////////////////////////////////////
    //
    // Filterbank class parameters
    //
    Filterbank* filterbank = dynamic_cast<Filterbank*>( operations[i].get() );

    if (filterbank) {

      dspR->set_nchan ( filterbank->get_nchan() );
      dspR->set_freq_res ( filterbank->get_freq_res() );
      dspR->set_time_res ( filterbank->get_time_res() );

    }

    // ////////////////////////////////////////////////////////////////////
    //
    // Convolution class parameters
    //

    Convolution* convolution = dynamic_cast<Convolution*>(operations[i].get());

    if (convolution) {

      if ( !convolution->has_response() )
        cerr << "dsp::Archiver::set Pulsar::dspReduction Convolution\n   "
             << convolution->get_name() << " instance with no Response" << endl;

      else {

        if (verbose) cerr << "dsp::Archiver::set Pulsar::dspReduction "
                             "Convolution with Response" << endl;

        const Response* response = convolution->get_response ();

        unsigned nsamp_fft = response->get_ndat();
        unsigned nsamp_overlap_pos = response->get_impulse_pos ();
        unsigned nsamp_overlap_neg = response->get_impulse_neg ();

        const TimeSeries* input = convolution->get_input ();

        if (input->get_state() == Signal::Nyquist) {
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
    TScrunch* tscrunch = dynamic_cast<TScrunch*>( operations[i].get() );

    if (tscrunch)
      dspR->set_ScrunchFactor ( tscrunch->get_factor() );


    // ////////////////////////////////////////////////////////////////////
    //
    // PhaseSeries class parameters
    //

    if (profiles)
      dspR->set_scale ( profiles->get_scale() );
  }

}
catch (Error& error) {
  throw error += "dsp::Archiver::set Pulsar::dspReduction";
}


void dsp::Archiver::set (Pulsar::TwoBitStats* tbc) try {

  if (verbose)
    cerr << "dsp::Archiver::set Pulsar::TwoBitStats Extension" << endl;

  if (!twobit) {
    if (verbose)
      cerr << "dsp::Archiver::set Pulsar::TwoBitStats no TwoBitCorrection"
	   << endl;
    return;
  }

  unsigned ndig = twobit->get_ndig ();
  unsigned ndat_per_weight = twobit->get_ndat_per_weight ();

  tbc->resize (ndat_per_weight, ndig);

  tbc->set_threshold ( twobit->get_threshold() );
  tbc->set_cutoff_sigma ( twobit->get_cutoff_sigma() );

  // temporary space
  vector<float> histogram;

  for (unsigned idig=0; idig<ndig; idig++)
  {
    twobit->get_histogram (histogram, idig);
    tbc->set_histogram (histogram, idig);
  }

}
catch (Error& error) {
  throw error += "dsp::Archiver::set Pulsar::TwoBitStats";
}


void dsp::Archiver::set (Pulsar::Passband* pband) try {

  if (verbose)
    cerr << "dsp::Archiver::set Pulsar::Passband Extension" << endl;

  if (!passband) {
    if (verbose)
      cerr << "dsp::Archiver::set Pulsar::Passband no passband" << endl;
    return;
  }

  // terminology differs between dsp::Shape and the Pulsar::Passband Extension
  unsigned npol = passband->get_npol ();
  unsigned nband = passband->get_nchan ();
  unsigned nchan = passband->get_ndat ();

  if (npol==0 || nband==0 || nchan==0) {
    if (verbose)
      cerr << "dsp::Archiver::set Pulsar::Passband empty passband" << endl;
    return;
  }

  dsp::Response copy (*passband);
  copy.naturalize ();
  
  if (verbose) cerr << "dsp::Archiver::set Pulsar::Passband Extension copy\n"
		 "  npol=" << npol << " nband=" << nband <<
		 " nchan/band=" << nchan << endl;
  
  pband->resize (nchan, npol, nband);
  
  if (passband->get_ndim() != 1)
    throw Error (InvalidState, "dsp::Archiver::set_passband",
		 "Passband Response ndim != 1");
  
  for (unsigned ipol=0; ipol<npol; ipol++)
    for (unsigned iband=0; iband<nband; iband++)
      pband->set_passband (copy.get_datptr (iband, ipol), ipol, iband);
  
}
catch (Error& error) {
  throw error += "dsp::Archiver::set Pulsar::Passband";
}


