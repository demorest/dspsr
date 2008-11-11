/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ObservationChange.h"

dsp::ObservationChange::ObservationChange ()
{
  telescope_changed = false;
  receiver_changed = false;
  source_changed = false;
  centre_frequency_changed = false;
  bandwidth_changed = false;
  type_changed = false;
  state_changed = false;
  basis_changed = false;
  rate_changed = false;
  start_time_changed = false;
  scale_changed = false;
  swap_changed = false;
  dc_centred_changed = false;
  identifier_changed = false;
  mode_changed = false;
  machine_changed = false;
  coordinates_changed = false;
  dispersion_measure_changed = false;
  ndat_changed = false;
  ndim_changed = false;
  nchan_changed = false;
  npol_changed = false;
  nbit_changed = false;
  calfreq_changed = false;
  dual_sideband_changed = false;
}

//! Set the telescope name
void dsp::ObservationChange::set_telescope (const std::string& arg)
{
  telescope_changed = true;
  Observation::set_telescope (arg);
}

//! Set the receiver name
void dsp::ObservationChange::set_receiver (const std::string& arg)
{
  receiver_changed = true;
  Observation::set_receiver (arg);
}

//! Set the source name
void dsp::ObservationChange::set_source (const std::string& arg)
{
  source_changed = true;
  Observation::set_source (arg);
}

//! Set the coordinates of the source
void dsp::ObservationChange::set_coordinates (sky_coord arg)
{
  coordinates_changed = true;
  Observation::set_coordinates (arg);
}

//! Set the centre frequency of the band-limited signal in MHz
void dsp::ObservationChange::set_centre_frequency (double arg)
{
  centre_frequency_changed = true;
  Observation::set_centre_frequency (arg);
}

//! Set the bandwidth of signal in MHz
void dsp::ObservationChange::set_bandwidth (double arg)
{
  bandwidth_changed = true;
  Observation::set_bandwidth (arg);
}

//! Set the start time of the leading edge of the first time sample
void dsp::ObservationChange::set_start_time (MJD arg)
{
  start_time_changed = true;
  Observation::set_start_time (arg);
}

//! Set the sampling rate (time samples per second in Hz)
void dsp::ObservationChange::set_rate (double arg)
{
  rate_changed = true;
  Observation::set_rate (arg);
}

//! Set the amount by which data has been scaled
void dsp::ObservationChange::set_scale (double arg)
{
  scale_changed = true;
  Observation::set_scale (arg);
}

//! Set true if frequency channels are out of order (band swappped)
void dsp::ObservationChange::set_swap (bool arg)
{
  swap_changed = true;
  Observation::set_swap (arg);
}

//! Set true if the data are dual sideband
void dsp::ObservationChange::set_dual_sideband (bool arg)
{
  dual_sideband_changed = true;
  Observation::set_dual_sideband (arg);
}

//! Set true if centre channel is centred on centre frequency
void dsp::ObservationChange::set_dc_centred (bool arg)
{
  dc_centred_changed = true;
  Observation::set_dc_centred (arg);
}

//! Set the observation identifier
void dsp::ObservationChange::set_identifier (const std::string& arg)
{
  identifier_changed = true;
  Observation::set_identifier (arg);
}

//! Set the instrument used to record signal
void dsp::ObservationChange::set_machine (const std::string& arg)
{
  machine_changed = true;
  Observation::set_machine (arg);
}

//! Set the dispersion measure recorded in the archive
void dsp::ObservationChange::set_dispersion_measure (double arg)
{
  dispersion_measure_changed = true;
  Observation::set_dispersion_measure (arg);
}

//! Set the rotation measure recorded in the archive
void dsp::ObservationChange::set_rotation_measure (double arg)
{
  rotation_measure_changed = true;
  Observation::set_rotation_measure (arg);
}

//! Set the observation mode
void dsp::ObservationChange::set_mode (const std::string& arg)
{
  mode_changed = true;
  Observation::set_mode (arg);
}

//! Set the type of receiver feeds
void dsp::ObservationChange::set_basis (Signal::Basis arg)
{
  basis_changed = true;
  Observation::set_basis (arg);
}

//! Set the state of the signal
void dsp::ObservationChange::set_state (Signal::State arg)
{
  state_changed = true;
  Observation::set_state (arg);
}

//! Set the source type
void dsp::ObservationChange::set_type (Signal::Source arg)
{
  type_changed = true;
  Observation::set_type (arg);
}

//! Set the dimension of each datum
void dsp::ObservationChange::set_ndim (unsigned arg)
{
  ndim_changed = true;
  Observation::set_ndim (arg);
}

//! Set the number of channels into which the band is divided
void dsp::ObservationChange::set_nchan (unsigned arg)
{
  nchan_changed = true;
  Observation::set_nchan (arg);
}

//! Set the number of polarizations
void dsp::ObservationChange::set_npol (unsigned arg)
{
  npol_changed = true;
  Observation::set_npol (arg);
}

//! Set the number of bits per value
void dsp::ObservationChange::set_nbit (unsigned arg)
{
  nbit_changed = true;
  Observation::set_nbit (arg);
}

void dsp::ObservationChange::set_ndat (uint64 arg)
{
  ndat_changed = true;
  Observation::set_ndat (arg);
}

//! Set the cal frequency
void dsp::ObservationChange::set_calfreq (double arg)
{
  calfreq_changed = true;
  Observation::set_calfreq (arg);
}
