#include "dsp/Archiver.h"
#include "dsp/Response.h"
#include "dsp/PhaseSeries.h"

#include "dsp/Input.h"
#include "dsp/TwoBitCorrection.h"
#include "dsp/Filterbank.h"
#include "dsp/Convolution.h"
#include "dsp/TScrunch.h"

#include "Pulsar/dspReduction.h"
#include "Pulsar/TwoBitStats.h"
#include "Pulsar/Passband.h"

#include "Error.h"


void dsp::Archiver::set (Pulsar::dspReduction* dspR)
{ try {

  if (verbose)
    cerr << "dsp::Archiver::set Pulsar::dspReduction Extension" << endl;

  if (!operations.size()) {
    if (verbose)
      cerr << "dsp::Archiver::set Pulsar::dspReduction no operations" << endl;
    return;
  }

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
    TwoBitCorrection* tbc;
    tbc = dynamic_cast<TwoBitCorrection*>( operations[i].get() );

    // save it for the TwoBitStats Extension
    if (tbc)
      twobit = tbc;


    // ////////////////////////////////////////////////////////////////////
    //
    // Filterbank class parameters
    //
    Filterbank* filterbank = dynamic_cast<Filterbank*>( operations[i].get() );
    Convolution* convolution = 0;

    if (filterbank) {

      dspR->set_nchan ( filterbank->get_nchan() );
      dspR->set_freq_res ( filterbank->get_freq_res() );
      dspR->set_time_res ( filterbank->get_time_res() );

      if ( filterbank->has_response() )
	convolution = filterbank;
    }

    // ////////////////////////////////////////////////////////////////////
    //
    // Convolution class parameters
    //
    if (!convolution)
      convolution = dynamic_cast<Convolution*>( operations[i].get() );

    if (convolution) {

      const Response* response = convolution->get_response ();

      unsigned nsamp_fft = response->get_ndat();
      unsigned nsamp_overlap_pos = response->get_impulse_pos ();
      unsigned nsamp_overlap_neg = response->get_impulse_neg ();

      TimeSeries* input = convolution->get_input ();

      if (input->get_state() == Signal::Nyquist) {
	nsamp_fft *= 2;
	nsamp_overlap_pos *= 2;
	nsamp_overlap_neg *= 2;
      }

      dspR->set_nsamp_fft ( nsamp_fft );
      dspR->set_nsamp_overlap_pos ( nsamp_overlap_pos );
      dspR->set_nsamp_overlap_neg ( nsamp_overlap_neg );

      // save it for the Passband Extension
      if ( convolution->has_passband() )
	passband = convolution->get_passband();

    }



    // ////////////////////////////////////////////////////////////////////
    //
    // Tscrunch class parameters
    //
    TScrunch* tscrunch = dynamic_cast<TScrunch*>( operations[i].get() );

    if (tscrunch) {

      dspR->set_ScrunchFactor ( tscrunch->get_ScrunchFactor() );

    }


    // ////////////////////////////////////////////////////////////////////
    //
    // PhaseSeries class parameters
    //

    if (profiles)
      dspR->set_scale ( profiles->get_scale() );
  }

}catch (Error& error) {
  throw error += "dsp::Archiver::set Pulsar::dspReduction";
}
}

void dsp::Archiver::set (Pulsar::TwoBitStats* tbc)
{ try {

  if (verbose)
    cerr << "dsp::Archiver::set Pulsar::TwoBitStats Extension" << endl;

  if (!twobit) {
    if (verbose)
      cerr << "dsp::Archiver::set Pulsar::TwoBitStats no TwoBitCorrection"
	   << endl;
    return;
  }

  //tbc->resize (nchan, npol, nband);

  //tbc->set_histogram (twobit->get_datptr (iband, ipol), ipol, iband);

}
catch (Error& error) {
  throw error += "dsp::Archiver::set Pulsar::Passband";
}
}


void dsp::Archiver::set (Pulsar::Passband* pband)
{ try {

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

  pband->resize (nchan, npol, nband);

  if (passband->get_ndim() != 1)
    throw Error (InvalidState, "dsp::Archiver::set_passband",
		 "Passband Response ndim != 1");

  for (unsigned ipol=0; ipol<npol; ipol++)
    for (unsigned iband=0; iband<nband; iband++)
      pband->set_passband (passband->get_datptr (iband, ipol), ipol, iband);

}
catch (Error& error) {
  throw error += "dsp::Archiver::set Pulsar::Passband";
}
}


