/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PrestoObservation.h"
#include "strutil.h"

using namespace std;

dsp::PrestoObservation::PrestoObservation (const infodata* data, 
					   unsigned extern_nbit)
{
  if (!data)
    throw Error (InvalidState, "dsp::PrestoObservation ctor",
		 "no infodata struct provided");

  telescope = data->telescope;
  receiver = "unknown";

  // remove spaces from source name
  source = remove_all (data->object, ' ');

  /*
    For an even number of channels, the centre frequency sits on the edge
    between the two central channels.  For an odd number of channels, the
    centre frequency is in the middle of the central channel.
  */
  centre_frequency = data->freq + double(data->num_chan - 1)*data->freqband/2;
  bandwidth = data->freqband;

  set_nchan (data->num_chan);
  set_nbit (extern_nbit);

  type = Signal::Pulsar;     // best estimate
  state = Signal::Intensity; // only supported option
  basis = Signal::Linear;    // doesn't matter

  rate = 1/data->dt;

  start_time = MJD (data->mjd_i, data->mjd_f);

  mode = data->band;
  machine = data->instrument;

  coordinates.ra().setHMS (data->ra_h, data->ra_m, data->ra_s);
  coordinates.dec().setDMS (data->dec_d, data->dec_m, data->dec_s);

  dispersion_measure = data->dm;
}
