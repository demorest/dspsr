#include <stdio.h>
#include <math.h>

#include <string>

#include "dsp/Observation.h"
#include "dsp/Telescope.h"

#include "MJD.h"

#include "genutil.h"
#include "string_utils.h"
#include "Error.h"

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

  telescope = 7;
  source = "unknown";
  centre_frequency = 0.0;
  bandwidth = 0.0;

  rate = 0.0;
  start_time = 0.0;
  scale = 1.0;

  swap = dc_centred = false;

  identifier = mode = machine = "";
  coordinates = sky_coord();
  dispersion_measure = 0.0;
}


void dsp::Observation::set_state (Signal::State _state)
{
  state = _state;

  if (state == Signal::Nyquist)
    ndim = 1;
  else if (state == Signal::Analytic)
    ndim = 2;
  else if (state == Signal::Intensity){
    ndim = 1;
    npol = 1;
  }
  else if (state == Signal::PPQQ){
    ndim = 1;
    npol = 2;
  }
  else if (state == Signal::Coherence){
    /* best not to muck with kludges */
  }
  else if (state == Signal::Stokes){
    /* best not to muck with kludges */
  }
  else if (state == Signal::Invariant){
    ndim = 1;
    npol = 1;
  }
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

/* this returns a flag that is true if the Observations may be combined 
   It doesn't check the start times- you have to do that yourself!
*/
bool dsp::Observation::combinable (const Observation & obs) const
{
  double eps = 0.000001;
  bool can_combine = true;

  if (telescope != obs.telescope){
    cerr << "dsp::Observation::combinable different telescope:"
	 << telescope << " and " << obs.telescope << endl;
  can_combine = false; }
  if (source != obs.source){
    cerr << "dsp::Observation::combinable different source:"
	 << source << " and " << obs.source << endl;
  can_combine = false; }
  if( fabs(centre_frequency-obs.centre_frequency) > eps ){
    cerr << "dsp::Observation::combinable different frequencies:"
	 << centre_frequency << " and " << obs.centre_frequency << endl;
  can_combine = false; }
  if( fabs(bandwidth-obs.bandwidth) > eps ){
  cerr << "dsp::Observation::combinable different bandwidth:"
	 << bandwidth << " and " << obs.bandwidth << endl;
  can_combine = false; }
  if (nchan != obs.nchan){
    cerr << "dsp::Observation::combinable different nchan:"
	 << nchan << " and " << obs.nchan << endl;
  can_combine = false; }
  if (npol != obs.npol){
    cerr << "dsp::Observation::combinable different npol:"
	 << npol << " and " << obs.npol << endl;
  can_combine = false; }
  if (ndim != obs.ndim){
    cerr << "dsp::Observation::combinable different ndim:"
	 << ndim << " and " << obs.ndim << endl;
  can_combine = false; }
  if (nbit != obs.nbit){
    cerr << "dsp::Observation::combinable different nbit:"
	 << nbit << " and " << obs.nbit << endl;
  can_combine = false; }
  if (type != obs.type){
    cerr << "dsp::Observation::combinable different type:"
	 << type << " and " << obs.type << endl;
  can_combine = false; }
  if (state != obs.state){
    cerr << "dsp::Observation::combinable different state:"
	 << state << " and " << obs.state << endl;
  can_combine = false; }
  if (basis != obs.basis){
    cerr << "dsp::Observation::combinable different feeds:"
	 << basis << " and " << obs.basis << endl;
  can_combine = false; }
  if( fabs(rate-obs.rate)/rate > 0.01 ){ /* ie must be within 1% */
    cerr << "dsp::Observation::combinable different rate:"
	 << rate << " and " << obs.rate << endl;
  can_combine = false; }
  if( fabs(scale-obs.scale) > eps ){
    cerr << "dsp::Observation::combinable different scale:"
	 << scale << " and " << obs.scale << endl;
  can_combine = false; }
  if (swap != obs.swap){
    cerr << "dsp::Observation::combinable different swap:"
	 << swap << " and " << obs.swap << endl;
  can_combine = false; }
  if (dc_centred != obs.dc_centred){
    cerr << "dsp::Observation::combinable different dc_centred:"
	 << dc_centred << " and " << obs.dc_centred << endl;
  can_combine = false; }
  if (identifier != obs.identifier){
    cerr << "dsp::Observation::combinable different identifier:"
	 << identifier << " and " << obs.identifier << endl;
  can_combine = false; }
  if (mode != obs.mode){
    cerr << "dsp::Observation::combinable different mode:"
	 << mode << " and " << obs.mode << endl;
  can_combine = false; }
  if (machine != obs.machine){
    cerr << "dsp::Observation::combinable different machine:"
	 << machine << " and " << obs.machine << endl;
  can_combine = false; }

  return can_combine;
}

bool dsp::Observation::contiguous (const Observation & obs) const
{
  double difference = (get_end_time() - obs.get_start_time()).in_seconds();

  return ( combinable(obs) && fabs(difference) < 1e3/rate );
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
    throw Error (InvalidState, "Observation::set_default_basis",
		 "unrecognized telescope: %c\n", telescope);
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

  set_centre_frequency ( in_obs.get_centre_frequency() );
  set_bandwidth   ( in_obs.get_bandwidth() );
  set_nchan       ( in_obs.get_nchan() );
  set_npol        ( in_obs.get_npol() );
  set_ndat        ( in_obs.get_ndat() );
  set_ndim        ( in_obs.get_ndim() );
  set_nbit        ( in_obs.get_nbit() );
  set_state       ( in_obs.get_state() );
  set_basis       ( in_obs.get_basis() );

  set_start_time  ( in_obs.get_start_time() );

  set_rate        ( in_obs.get_rate() );
  set_scale       ( in_obs.get_scale() );
  set_swap        ( in_obs.get_swap() );
  set_dc_centred  ( in_obs.get_dc_centred() );

  set_telescope   ( in_obs.get_telescope() );

  set_source      ( in_obs.get_source() );
  set_identifier  ( in_obs.get_identifier() );
  set_machine     ( in_obs.get_machine() );
  set_mode        ( in_obs.get_mode() );

  set_coordinates ( in_obs.get_coordinates() );

  return *this;
}

// returns the centre_frequency of the ichan channel
double dsp::Observation::get_centre_frequency (unsigned ichan) const
{
  double chan_bandwidth = bandwidth / double(nchan);

  double lower_cfreq = 0.0;

  if (dc_centred)
    lower_cfreq = centre_frequency - bandwidth/2.0;
  else
    lower_cfreq = centre_frequency - 0.5*(double(bandwidth)-chan_bandwidth);

  unsigned swap_chan = 0;
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

//! Returns all information contained in this class into the string info_string
bool dsp::Observation::retrieve(string& ss){
  char dummy[300];

  sprintf(dummy,"NDAT\t"I64"\n",ndat); ss += dummy;
  sprintf(dummy,"TELESCOPE\t%c\n",telescope); ss += dummy;
  sprintf(dummy,"SOURCE\t%s\n",source.c_str()); ss += dummy;
  sprintf(dummy,"CENTRE_FREQUENCY\t%.16f\n",centre_frequency); ss += dummy;
  sprintf(dummy,"BANDWIDTH\t%.16f\n",bandwidth); ss += dummy;
  sprintf(dummy,"NCHAN\t%d\n",nchan); ss += dummy;
  sprintf(dummy,"NPOL\t%d\n",npol); ss += dummy;
  sprintf(dummy,"NDIM\t%d\n",ndim); ss += dummy;
  sprintf(dummy,"NBIT\t%d\n",nbit); ss += dummy;
  sprintf(dummy,"TYPE\t%s\n",Signal::source_string(type)); ss += dummy;
  sprintf(dummy,"STATE\t%s\n",Signal::state_string(state)); ss += dummy;
  sprintf(dummy,"BASIS\t%s\n",Signal::basis_string(basis)); ss += dummy;
  sprintf(dummy,"RATE\t%.16f\n",rate); ss += dummy;
  sprintf(dummy,"START_TIME\t%s\n",start_time.printall()); ss += dummy;
  sprintf(dummy,"SCALE\t%.16f\n",scale); ss += dummy;
  sprintf(dummy,"SWAP\t%s\n",swap?"true":"false"); ss += dummy;
  sprintf(dummy,"DC_CENTRED\t%s\n",dc_centred?"true":"false"); ss += dummy;
  sprintf(dummy,"IDENTIFIER\t%s\n",identifier.c_str()); ss += dummy;
  sprintf(dummy,"MODE\t%s\n",mode.c_str()); ss += dummy;
  sprintf(dummy,"MACHINE\t%s\n",machine.c_str()); ss += dummy;
  sprintf(dummy,"DISPERSION_MEASURE\t%.16f\n",dispersion_measure); ss += dummy;


  sprintf(dummy,"RAJ\t%s\n",coordinates.ra().getHMS().c_str()); ss += dummy;
  sprintf(dummy,"DECJ\t%s\n",coordinates.dec().getDMS().c_str()); ss += dummy;

  return true;
}
    
//! Writes all information contained in this class into the fptr at the current file offset
bool dsp::Observation::retrieve(FILE* fptr){
  string ss;
  if( !retrieve(ss) ){
    fprintf(stderr,"dsp::Observation::retrieve() failed to write to fptr because string version failed\n");
    fclose(fptr);
    return false;
  }

  fprintf(fptr,"%s",ss.c_str());

  return true;
}
