/***************************************************************************
 *
 *   Copyright (C) 2002-2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Observation.h"
#include "dsp/ObservationInterface.h"

#include "Error.h"
#include "dirutil.h"

#include <stdio.h>
#include <math.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

using namespace std;

bool dsp::Observation::verbose = false;

dsp::Observation::Observation ()
{
  init ();
}

void dsp::Observation::init ()
{
  ndat = 0;
  nchan = 1;
  npol = 1;
  ndim = 1;
  nbit = 0;

  type = Signal::Pulsar;
  state = Signal::Intensity;
  basis = Signal::Linear;

  telescope = "unknown";
  receiver = "unknown";
  source = "unknown";

  centre_frequency = 0.0;
  bandwidth = 0.0;
  calfreq = 0.0;

  rate = 0.0;
  start_time = 0.0;
  scale = 1.0;

  swap = dc_centred = false;
  nsub_swap = 0;

  identifier = mode = machine = "";
  coordinates = sky_coord();
  dispersion_measure = 0.0;
  rotation_measure = 0.0;

  dual_sideband = -1;
  require_equal_sources = true;
  require_equal_rates = true;
}

//! Set true if the data are dual sideband
void dsp::Observation::set_dual_sideband (bool dual)
{
  // not sure how bool casts to char, so making it explicit
  if (dual)
    dual_sideband = 1;
  else
    dual_sideband = 0;
}


//! Return true if the data are dual_sideband
bool dsp::Observation::get_dual_sideband () const
{
  if (dual_sideband != -1)
    return (dual_sideband == 1);

  // if the dual sideband flag is not set, return true if state == Analytic
  return state == Signal::Analytic;
}


void dsp::Observation::set_state (Signal::State _state)
{
  state = _state;

  if (state == Signal::Nyquist)
    set_ndim(1);
  else if (state == Signal::Analytic)
    set_ndim(2);
  else if (state == Signal::Intensity){
    set_ndim(1);
    set_npol( 1 );
  }
  else if (state == Signal::PPQQ){
    set_ndim(1);
    set_npol( 2 );
  }
  else if (state == Signal::PP_State || state==Signal::QQ_State){
    set_ndim( 1 );
    set_npol( 1 );
  }
  else if (state == Signal::Coherence){
    /* best not to muck with kludges */
  }
  else if (state == Signal::Stokes){
    /* best not to muck with kludges */
  }
  else if (state == Signal::Invariant){
    set_ndim(1);
    set_npol( 1 );
  }
}

/*! 
  \retval boolean true if the state of the Observation is valid
  \param reason If the return value is false, a string describing why
*/
bool dsp::Observation::state_is_valid (string& reason) const
{
  return Signal::valid_state(get_state(),get_ndim(),get_npol(),reason);
}
  
bool dsp::Observation::get_detected () const
{
  return (state != Signal::Nyquist && state != Signal::Analytic);
}

/* this returns a flag that is true if the Observations may be combined 
   It doesn't check the start times- you have to do that yourself!
*/
bool dsp::Observation::combinable (const Observation & obs) const
{
  bool can_combine = true;
  double eps = 0.000001;

  reason = "";
  string separator = "\n\t";

  if (telescope != obs.telescope)
  {
    reason += separator +
	"different telescopes:" + telescope + " != " + obs.telescope;
    can_combine = false;
  }

  if (receiver != obs.receiver)
  {
    reason += separator +
	"different receivers:" + receiver + " != " + obs.receiver;
    can_combine = false;
  }

  if (require_equal_sources && source != obs.source)
  {
    reason += separator +
	"different sources:" + source + " != " + obs.source;
    can_combine = false;
  }

  if (fabs(centre_frequency-obs.centre_frequency) > eps)
  {
    reason += separator +
	"different centre frequencies:" + tostring(centre_frequency) + 
         " != " + tostring(obs.centre_frequency);
    can_combine = false;
  }

  else if( fabs(bandwidth-obs.bandwidth) > eps )
  {
    reason += separator +
	"different bandwidths:" + tostring(bandwidth) + 
         " != " + tostring(obs.bandwidth);
    can_combine = false;
  }

  if (nchan != obs.nchan)
  {
    reason += separator +
	"different nchans:" + tostring(nchan) + " != " + tostring(obs.nchan);
    can_combine = false;
  }
 
  if (npol != obs.npol)
  {
    reason += separator +
	"different npols:" + tostring(npol) + " != " + tostring(obs.npol);
    can_combine = false;
  }

  if (ndim != obs.ndim)
  {
    reason += separator +
	"different ndims:" + tostring(ndim) + " != " + tostring(obs.ndim);
    can_combine = false;
  }

  if (nbit != obs.nbit)
  {
    reason += separator +
	"different nbits:" + tostring(nbit) + " != " + tostring(obs.nbit);
    can_combine = false;
  }

  if (type != obs.type)
  {
    reason += separator +
	"different npols:" + tostring(npol) + " != " + tostring(obs.npol);
    can_combine = false;
  }

  if (state != obs.state)
  {
    reason += separator +
	"different states:" + tostring(state) + " != " + tostring(obs.state);
    can_combine = false;
  }

  if (basis != obs.basis)
  {
    reason += separator +
	"different bases:" + tostring(basis) + " != " + tostring(obs.basis);
    can_combine = false;
  }
  
  if (require_equal_rates && rate != obs.rate)
  {
    reason += separator +
	"different rates:" + tostring(rate) + " != " + tostring(obs.rate);
    can_combine = false;
  }
  
  if ( fabs(scale-obs.scale) > eps*fabs(scale) )
  {
    reason += separator +
	"different scales:" + tostring(scale) + " != " + tostring(obs.scale);
    can_combine = false;
  }
  
  if (swap != obs.swap)
  {
    reason += separator +
	"different swaps:" + tostring(swap) + " != " + tostring(obs.swap);
    can_combine = false;
  }
  
  if (nsub_swap != obs.nsub_swap)
  {
    reason += separator +
	"different nsub_swaps:" + tostring(swap) + " != " + tostring(obs.swap);
    can_combine = false;
  }

  if (dc_centred != obs.dc_centred)
  {
    reason += separator +
	"different dccs:" + tostring(dc_centred) +
        " != " + tostring(obs.dc_centred);
    can_combine = false;
  }
  
  if (mode != obs.mode )
  {
    string s1 = mode.substr(0,5);
    string s2 = obs.mode.substr(0,5);

    if (!(s1==s2 && s1=="2-bit"))
    {
      reason += separator +
	"different modes:" + tostring(mode) + " != " + tostring(obs.mode);
      can_combine = false;
    }
  }
  
  if (machine != obs.machine)
  {
    reason += separator +
	"different machines:" + tostring(machine) + " != " + tostring(obs.machine);
    can_combine = false;
  }
  
  if( fabs(dispersion_measure - obs.dispersion_measure) > eps)
  {
    reason += separator +
	"different dispersion measures:" + tostring(dispersion_measure) +
        " != " + tostring(obs.dispersion_measure);
    can_combine = false;
  }
  
  if( fabs(rotation_measure - obs.rotation_measure) > eps)
  {
    reason += separator +
	"different rotation measures:" + tostring(rotation_measure) + 
        " != " + tostring(obs.rotation_measure);
    can_combine = false;
  }

  return can_combine;
}

bool dsp::Observation::contiguous (const Observation & obs) const
{
  if (verbose)
    cerr << "dsp::Observation::contiguous this=" << this << " obs=" << &obs
         << endl;

  double difference = (obs.get_start_time() - get_end_time()).in_seconds();

  if (verbose)
    cerr << "dsp::Observation::contiguous difference=" << difference
         << "s rate=" << rate << "Hz" << endl;

  bool combinable = obs.combinable (*this);
  bool contiguous = fabs(difference) < 0.9/rate;

#if 0
  if ( !contiguous && verbose_on_failure ) {

    cerr << "dsp::Observation::contiguous returning false:\n\t"
      "this.start_time=" << get_start_time() << "\n\t"
      "this.end_time  =" << get_end_time() << "\n\t"
      "that.start_time=" << obs.get_start_time() << "\n\t"
      "difference=" << difference*1e6 << "us "
      "needed to be less than " << 0.9e6/rate << "us.\n\t"
      "At sampling rate=" << rate/1e6 << "MHz, that.start_time is off by "
         << difference * rate << " samples" << endl;

  } 
#endif

  if (verbose)
    cerr << "dsp::Observation::contiguous return" << endl;

  return contiguous && combinable;
}

dsp::Observation::~Observation()
{
}

#if 1
dsp::Observation::Observation (const Observation & in_obs)
{
  init ();
  dsp::Observation::operator=(in_obs);
}

//! Copy the dimensions of another observation
void dsp::Observation::copy_dimensions (const Observation* other)
{
  set_state( other->get_state() );
  set_ndim ( other->get_ndim() );
  set_nchan( other->get_nchan() );
  set_npol ( other->get_npol() );
  set_nbit ( other->get_nbit() );
  set_ndat ( other->get_ndat() );
}

const dsp::Observation& dsp::Observation::operator = (const Observation& in_obs)
{
  if (this == &in_obs)
    return *this;

  set_basis       ( in_obs.get_basis() );
  set_state       ( in_obs.get_state() );
  set_type        ( in_obs.get_type() );

  set_ndim        ( in_obs.get_ndim() );
  set_nchan       ( in_obs.get_nchan() );
  set_npol        ( in_obs.get_npol() );
  set_nbit        ( in_obs.get_nbit() );
  set_ndat        ( in_obs.get_ndat() );

  set_telescope   ( in_obs.get_telescope() );
  set_receiver    ( in_obs.get_receiver() );
  set_source      ( in_obs.get_source() );
  set_coordinates ( in_obs.get_coordinates() );

  dual_sideband         = in_obs.dual_sideband;

  set_centre_frequency   ( in_obs.get_centre_frequency() );
  set_bandwidth          ( in_obs.get_bandwidth() );
  set_dispersion_measure ( in_obs.get_dispersion_measure() );
  set_rotation_measure   ( in_obs.get_rotation_measure() );

  set_start_time  ( in_obs.get_start_time() );

  set_rate        ( in_obs.get_rate() );
  set_scale       ( in_obs.get_scale() );
  set_swap        ( in_obs.get_swap() );
  set_nsub_swap   ( in_obs.get_nsub_swap() );
  set_dc_centred  ( in_obs.get_dc_centred() );

  set_identifier  ( in_obs.get_identifier() );
  set_machine     ( in_obs.get_machine() );
  set_mode        ( in_obs.get_mode() );
  set_calfreq     ( in_obs.get_calfreq());

  return *this;
}

#endif

unsigned dsp::Observation::get_unswapped_ichan (unsigned ichan) const
{
  if (swap)
    ichan = (ichan + get_nchan()/2) % get_nchan();

  if (nsub_swap)
  {
    unsigned sub_nchan = get_nchan() / nsub_swap;
    div_t result = div ( (int)ichan, (int)sub_nchan );
    unsigned sub_band = result.quot;
    unsigned sub_ichan = (result.rem + sub_nchan/2) % sub_nchan;
    ichan = sub_band * sub_nchan + sub_ichan;
  }

  return ichan;
}

// returns the centre_frequency of the ichan channel
double dsp::Observation::get_centre_frequency (unsigned ichan) const
{
  double channel = get_unswapped_ichan (ichan);
  return get_base_frequency() + channel * bandwidth / double(get_nchan());
}

// returns the centre_frequency of the first channel
double dsp::Observation::get_base_frequency () const
{
  if (dc_centred)
    return centre_frequency - 0.5*bandwidth;
  else
    return centre_frequency - 0.5*bandwidth + 0.5*bandwidth/double(get_nchan());
}

//! Change the state and correct other attributes accordingly
void dsp::Observation::change_state (Signal::State new_state)
{
  if (new_state == Signal::Analytic && state == Signal::Nyquist) {
    /* Observation was originally single-sideband, Nyquist-sampled.
       Now it is complex, quadrature sampled */
    state = Signal::Analytic;
    set_ndat( get_ndat() / 2 );         // number of complex samples
    rate /= 2.0;       // samples are now complex at half the rate
    set_ndim(2);          // the dimension of each datum is now 2 [re+im]
  }

  state = new_state;
}

//! Change the start time by the number of time samples specified
void dsp::Observation::change_start_time (int64_t samples)
{
  start_time += double(samples)/rate;
}

//! Return the end time of the trailing edge of the last time sample
// Returns correct answer if ndat=rate=0 and avoids division by zero
MJD dsp::Observation::get_end_time () const
{
  if( ndat==0 )
    return start_time;
  return start_time + double(ndat) / rate;
}

TextInterface::Parser* dsp::Observation::get_interface ()
{
  return new Interface (this);
}
