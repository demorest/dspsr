#include <stdio.h>

#include "Observation.h"
#include "Telescope.h"

#include "genutil.h"
#include "string_utils.h"

bool dsp::Observation::verbose = false;

int64 dsp::Observation::verbose_nbytes (int64 nsamples) const
{
  fprintf(stderr,"nsamples="I64"\tnbit="I64"\tnpol="I64"\tnchan="I64"\tget_ndim()="I64"\n",
	  int64(nsamples),
	  int64(nbit),
	  int64(npol),
	  int64(nchan),
	  int64(get_ndim()));
  fprintf(stderr,"\t"I64"\t"I64"\t"I64"\t"I64"\t"I64"\t"I64"\n",
	  int64(nsamples),
	  int64(nsamples*nbit),
	  int64(nsamples*nbit*npol),
	  int64(nsamples*nbit*npol*nchan),
	  int64(nsamples*nbit*npol*nchan*get_ndim()),
	  int64((nsamples*nbit*npol*nchan*get_ndim())/8));

  return (nsamples*nbit*npol*nchan*get_ndim())/8;
}

dsp::Observation::Observation ()
{
  init ();
}

void dsp::Observation::init ()
{
  ndat = 0;
  nchan = 1;
  npol = 1;
  ndim = -1;
  nbit = -1;

  centre_frequency = 0;
  bandwidth = 0;

  basis = Signal::Linear;
  state = Signal::Intensity;
  type = Signal::Pulsar;

  start_time = 0.0;
  rate = 0;

  scale = 1;

  swap = dc_centred = false;
  telescope = 0;
  source = identifier = mode = machine = "";
  position = sky_coord();
}

void dsp::Observation::set_sample (Signal::State _state,
				   int _nchan, int _npol,
				   int _ndim, int _nbit)
{
  // check state and dimension information
  state = _state;
  nchan = _nchan;
  npol = _npol;
  ndim = _ndim;
  nbit = _nbit;

  string reason;
  if (!state_is_valid (reason))
    throw_str ("Observation::set_sample invalid state: " + reason);
}

void dsp::Observation::set_state (Signal::State _state)
{
  state = _state;

  if (state == Signal::Nyquist)
    ndim = 1;
  else if (state == Signal::Analytic)
    ndim = 2;
}

/*! 
  \retval boolean true if the state of the Observation is valid
  \param reason If the return value is false, a string describing why
*/
bool dsp::Observation::state_is_valid (string& reason) const
{
  switch (state) {
  case Signal::Nyquist:
    if (ndim != 1)  {
      reason = "state=" + get_state_as_string() + " and ndim!=1";
      return false;
    }
    break;

  case Signal::Analytic:
    if (ndim != 2) {
      reason = "state=" + get_state_as_string() + " and ndim!=2";
      return false;
    }
    break;

  case Signal::Invariant:
  case Signal::Intensity:
    if (npol != 1) {
      reason = "state=" + get_state_as_string() + " and npol!=1";
      return false;
    }
    break;

  case Signal::PPQQ:
    if (npol != 2) {
      reason = "state=" + get_state_as_string() + " and npol!=2";
      return false;
    }
    break;

  case Signal::Coherence:
  case Signal::Stokes:
    if (ndim*npol != 4) {
      reason = "state=" + get_state_as_string() + " and ndim*npol!=4";
      return false;
    }
    break;

  default:
    reason = "unknown state";
    return false;
  }

  return true;
}


bool dsp::Observation::get_detected () const
{
  return (state != Signal::Nyquist && state != Signal::Analytic);
}

/* this returns a flag that is true if an Observation may be combined 
   with this object using operators like += and things like that */
bool dsp::Observation::combinable (const Observation & obs)
{
  if (verbose) {
    if (centre_frequency != obs.centre_frequency)
      cerr << "dsp::Observation::combinable different frequencies:"
	   << centre_frequency << " and " << obs.centre_frequency << endl;
    if (bandwidth != obs.bandwidth)
      cerr << "dsp::Observation::combinable different bandwidth:"
	   << bandwidth << " and " << obs.bandwidth << endl;
    if (nchan != obs.nchan)
      cerr << "dsp::Observation::combinable different nchan:"
	   << nchan << " and " << obs.nchan << endl;
    if (npol != obs.npol)
      cerr << "dsp::Observation::combinable different npol:"
	   << npol << " and " << obs.npol << endl;
    if (ndim != obs.ndim)
      cerr << "dsp::Observation::combinable different ndim:"
	   << ndim << " and " << obs.ndim << endl;
    if (state != obs.state)
      cerr << "dsp::Observation::combinable different state:"
	   << state << " and " << obs.state << endl;
    if (basis != obs.basis)
      cerr << "dsp::Observation::combinable different feeds:"
	   << basis << " and " << obs.basis << endl;
  }

  return ( (centre_frequency == obs.centre_frequency) &&
	   (bandwidth    == obs.bandwidth)    &&
	   (nchan == obs.nchan) &&
	   (npol  == obs.npol)  &&
	   (ndim  == obs.ndim)  &&
	   (state == obs.state) &&
	   (source == obs.source) &&
	   (swap  == obs.swap) &&
	   (dc_centred == obs.dc_centred) );
}

void dsp::Observation::set_telescope (char _telescope)
{
  if (_telescope < 10) /* if the char is < 10 then it was probably an int */
    _telescope += '0';
 
  telescope = _telescope;
}

void dsp::Observation::set_default_basis ()
{
  if (telescope == Telescope::Parkes)  {
    // Parkes has linear feeds for multibeam, H-OH, and 50cm
    basis = Signal::Linear;
    // above 2 GHz, can assume that the Galileo receiver is used
    if (centre_frequency > 2000.0)
      basis = Signal::Circular;
  }
  else if (telescope == Telescope::ATCA)
    basis = Signal::Circular;
  else if (telescope == Telescope::Tidbinbilla)
    basis = Signal::Circular;
  else if (telescope == Telescope::Arecibo)
    basis = Signal::Circular;
  else if (telescope == Telescope::Hobart)
    basis = Signal::Circular;
  else
    throw_str ("Observation::set_default_basis no info telid: %c\n",
	       telescope);
}

string dsp::Observation::get_default_id (const MJD& mjd)
{
  static char id [15];
  utc_t startutc = UTC_INIT;

  mjd.UTC (&startutc, NULL);
  utc2str (id, startutc, "yyyydddhhmmss");

  return string (id);
}

string dsp::Observation::get_default_id () const
{
  return get_default_id (start_time);
}

string dsp::Observation::get_state_as_string () const
{
  return Signal::state_string (state);
}

dsp::Observation::Observation (const Observation & in_obs)
{
  init ();
  dsp::Observation::operator=(in_obs);
}

dsp::Observation::~Observation()
{
}

dsp::Observation& dsp::Observation::operator = (const Observation& in_obs)
{
  if (this == &in_obs)
    return *this;

  ndat        = in_obs.ndat;
  centre_frequency = in_obs.centre_frequency;
  bandwidth   = in_obs.bandwidth;
  nchan       = in_obs.nchan;
  npol        = in_obs.npol;
  ndim        = in_obs.ndim;
  nbit        = in_obs.nbit;
  state       = in_obs.state;
  basis    = in_obs.basis;

  start_time  = in_obs.start_time;

  rate        = in_obs.rate;
  scale       = in_obs.scale;
  swap        = in_obs.swap;
  dc_centred  = in_obs.dc_centred;

  telescope   = in_obs.telescope;

  source      = in_obs.source;
  identifier  = in_obs.identifier;
  machine     = in_obs.machine;
  mode        = in_obs.mode;

  position    = in_obs.position;

  return *this;
}

// returns the centre_frequency of the ichan channel
double dsp::Observation::get_centre_frequency (int ichan) const
{
  double chan_bandwidth = bandwidth / double(nchan);

  double lower_cfreq = 0.0;

  if (dc_centred)
    lower_cfreq = centre_frequency - bandwidth/2.0;
  else
    lower_cfreq = centre_frequency - 0.5*(double(bandwidth)-chan_bandwidth);

  int swap_chan = 0;
  if (swap)
    swap_chan = nchan/2;

  double channel = double ( (ichan+swap_chan) % nchan );

  return lower_cfreq + channel * chan_bandwidth;
}

//! Change the state and correct other attributes accordingly
void dsp::Observation::change_state (Signal::State new_state)
{
  if (new_state == Signal::Analytic && state == Signal::Nyquist) {
    /* Observation was originally single-sideband, Signal::Nyquist-sampled data.
       Now it is complex, quadrature sampled */
    state = Signal::Analytic;
    ndat /= 2;         // number of complex samples
    rate /= 2.0;       // samples are now complex at half the rate
  }

  state = new_state;
}

//! Change the start time by the number of time samples specified
void dsp::Observation::change_start_time (int64 ndat)
{
  start_time += double(ndat)/rate;
}

