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
  set_telescope( string(1, '0' + telescope_id) );
  set_machine( string(1, '0' + machine_id) );

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

  // set_ndat (nsamples);

  set_state( Signal::Intensity );

  set_rate( 1.0/tsamp );
  set_start_time( tstart );

  sky_coord coord;
  coord.ra().setHourMS (src_raj);
  coord.dec().setDegMS (src_dej);

  set_coordinates (coord);
}

void dsp::SigProcObservation::unload (FILE* header)
{
  unload_global ();
  filterbank_header (header);
}

void dsp::SigProcObservation::unload_global ()
{
  telescope_id = get_telescope()[0] - '0';
  machine_id = get_machine()[0] - '0';

  // set_receiver (buffer);

  strcpy( source_name, get_source().c_str() );

  // set_centre_frequency (freq);
  // set_bandwidth (bw);

  nchans = get_nchan ();
  nifs = get_npol ();
  nbits = get_nbit ();
  
  tsamp = 1.0 / get_rate ();

  tstart = get_start_time().in_days();

  src_raj = coordinates.ra().getHourMS ();
  src_dej = coordinates.dec().getDegMS ();
}
