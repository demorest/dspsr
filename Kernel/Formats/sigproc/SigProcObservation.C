/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SigProcObservation.h"

extern "C" {
#include "sigproc.h"
#include "header.h"
}

using namespace std;

dsp::SigProcObservation::SigProcObservation (FILE* header)
{
  if (header)
    load (header);
}

void dsp::SigProcObservation::load (FILE* header)
{
  read_header (header);
  load_global ();
}

void dsp::SigProcObservation::load_global ()
{
  // no idea about the size of the data
  //
  set_ndat( 0 );

  set_telescope( string(1, char(telescope_id)) );

  // set_receiver (buffer);

  set_source( source_name );

  set_type( Signal::Pulsar );

  // set_calfreq(calfreq);

  // set_centre_frequency (freq);
  // set_bandwidth (bw);

  set_nchan (nchans);
  set_npol (nifs);
  set_nbit (nbits);
  set_ndim (1);

  set_state( Signal::Intensity );

  set_rate( 1.0/tsamp );
  set_start_time( mjdobs + tstart );
  set_machine( string(1, char(machine_id)) );
  coordinates.setRadians (src_raj, src_dej);
}

void dsp::SigProcObservation::unload (FILE* header)
{
  unload_global ();
  filterbank_header (header);
}

void dsp::SigProcObservation::unload_global ()
{
  telescope_id = get_telescope()[0];
  machine_id = get_machine()[0];

  // set_receiver (buffer);

  strcpy( source_name, get_source().c_str() );

  // set_centre_frequency (freq);
  // set_bandwidth (bw);

  nchans = get_nchan ();
  nifs = get_npol ();
  nbits = get_nbit ();
  
  tsamp = 1.0 / get_rate ();

  // get_start_time( mjdobs + tstart );

  // coordinates.setRadians (src_raj, src_dej);
}
