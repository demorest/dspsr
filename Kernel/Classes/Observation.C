#include "Observation.h"
#include "genutil.h"
#include "string_utils.h"

bool dsp::Observation::verbose = false;

dsp::Observation::Observation ()
{
  init ();
}

void dsp::Observation::init ()
{
  ndat = 0;
  centre_frequency = 0;
  bandwidth = 0;
  nchan = 1;
  npol = 1;
  feedtype = Invalid;
  start_time = 0.0;
  rate = 0;
  scale = 1;
  state = Unknown;
  swap = dc_centred = false;
  telescope = 0;
  source = identifier = mode = machine = "";
  position = sky_coord();
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
    if (state != obs.state)
      cerr << "dsp::Observation::combinable different state:"
	   << state << " and " << obs.state << endl;
    if (feedtype != obs.feedtype)
      cerr << "dsp::Observation::combinable different feeds:"
	   << feedtype << " and " << obs.feedtype << endl;
  }

  return ( (centre_frequency == obs.centre_frequency) &&
	   (bandwidth    == obs.bandwidth)    &&
	   (nchan == obs.nchan) &&
	   (npol  == obs.npol)  &&
	   (state == obs.state) &&
	   (source == obs.source) &&
	   (swap  == obs.swap) &&
	   (dc_centred == obs.dc_centred) );
}

void dsp::Observation::set_default_feedtype ()
{
  if (telescope == TELID_PKS)  {
    // Parkes has linear feeds for multibeam, H-OH, and 50cm
    feedtype = Linear;
    // above 2 GHz, can assume that the Galileo receiver is used
    if (centre_frequency > 2000.0)
      feedtype = Circular;
  }
  else if (telescope == TELID_ATCA)
    feedtype = Circular;
  else if (telescope == TELID_TID)
    feedtype = Circular;
  else if (telescope == TELID_ARECIBO)
    feedtype = Circular;
  else if (telescope == TELID_HOBART)
    feedtype = Circular;
  else if (telescope == TELID_HOBART)
    feedtype = Circular;
  else
    throw_str ("Observation::set_default_feedtype no info telid: %d\n", telescope);
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

string dsp::Observation::get_state_str () const
{
#define OBS_OPT(st) case st: return string (#st)
  // possible states of the data
  switch (state) { 
    OBS_OPT (Unknown);
    OBS_OPT (Nyquist);
    OBS_OPT (Analytic);
    OBS_OPT (Detected);
    OBS_OPT (Coherence);
    OBS_OPT (Stokes);
  };
  return string ("invalid");
#undef OBS_OPT
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
  nbit        = in_obs.nbit;
  state       = in_obs.state;
  feedtype    = in_obs.feedtype;

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
void dsp::Observation::change_state (State new_state)
{
  if (new_state == Analytic && state == Nyquist) {
    /* Observation was originally single-sideband, Nyquist-sampled data.
       Now it is complex, quadrature sampled */
    state = Analytic;
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
