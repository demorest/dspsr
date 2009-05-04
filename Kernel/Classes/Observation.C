/***************************************************************************
 *
 *   Copyright (C) 2002-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Observation.h"

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
  : cerr (std::cerr.rdbuf())
{
  init ();
}

//! Set verbosity ostream
void dsp::Observation::set_ostream (std::ostream& os) const
{
  this->cerr.rdbuf( os.rdbuf() );
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
bool dsp::Observation::combinable (const Observation & obs,
				   bool different_bands,
				   bool combinable_verbose, 
				   int ichan, int ipol) const
{
  bool can_combine = true;
  double eps = 0.000001;

  if (telescope != obs.telescope) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different telescope:"
	      << telescope << " and " << obs.telescope << endl;
    can_combine = false;
  }

  if (receiver != obs.receiver) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different receiver:"
	   << receiver << " and " << obs.receiver << endl;
    can_combine = false;
  }

  if (require_equal_sources && source != obs.source)
  {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different source:"
	   << source << " and " << obs.source << endl;
    can_combine = false;
  }

  if( !different_bands ){
    if( ichan<0 ){
      if( fabs(centre_frequency-obs.centre_frequency) > eps ) {
	if (verbose || combinable_verbose)
	  cerr << "dsp::Observation::combinable different frequencies:"
	       << centre_frequency << " and " << obs.centre_frequency << endl;
	can_combine = false;
      }
    }
    else{
      if( fabs(get_centre_frequency(ichan)-obs.centre_frequency) > eps ) {
	if (verbose || combinable_verbose)
	  cerr << "dsp::Observation::combinable different frequencies in channel"
	       << ichan << ":" << get_centre_frequency(ichan)
	       << " and " << obs.centre_frequency << endl;
	can_combine = false;
      }
    }
  }  

  if( ichan>=0 ){
    if( fabs(bandwidth/float(get_nchan()) - obs.bandwidth) > eps ) {
      if (verbose || combinable_verbose)
	cerr << "dsp::Observation::combinable different channel bandwidth:"
	     << bandwidth/float(get_nchan()) << " and " << obs.bandwidth << endl;
      can_combine = false;
    }
  }
  else if( fabs(bandwidth-obs.bandwidth) > eps ) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different bandwidth:"
	   << bandwidth << " and " << obs.bandwidth << endl;
    can_combine = false;
  }

  if (get_nchan() != obs.get_nchan() && ichan<0) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different nchan:"
	   << get_nchan() << " and " << obs.get_nchan() << endl;
    can_combine = false;
  }
 
  if (get_npol() != obs.get_npol() && ipol<0 ) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different npol:"
	   << get_npol() << " and " << obs.get_npol() << endl;
    can_combine = false;
  }

  if (get_ndim() != obs.get_ndim()) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different ndim:"
	   << get_ndim() << " and " << obs.get_ndim() << endl;
    can_combine = false;
  }

  if (get_nbit() != obs.get_nbit()) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different nbit:"
	   << get_nbit() << " and " << obs.get_nbit() << endl;
    can_combine = false;
  }

  if (type != obs.type) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different type:"
	   << type << " and " << obs.type << endl;
    can_combine = false;
  }

  if (state != obs.state) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different state:"
	   << state << " and " << obs.state << endl;
    can_combine = false;
  }

  if (basis != obs.basis) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different feeds:"
	   << basis << " and " << obs.basis << endl;
    can_combine = false;
  }
  
  if (require_equal_rates && rate != obs.rate) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different rate:"
	   << rate << " and " << obs.rate << endl;
    can_combine = false;
  }
  
  if( fabs(scale-obs.scale) > eps*fabs(scale) ) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different scale:"
	   << scale << " and " << obs.scale << endl;
    can_combine = false;
  }
  
  if (swap != obs.swap) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different swap:"
	   << swap << " and " << obs.swap << endl;
    can_combine = false;
  }
  
  if (dc_centred != obs.dc_centred && ichan<0) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different dc_centred:"
	   << dc_centred << " and " << obs.dc_centred << endl;
    can_combine = false;
  }
  
  if (mode != obs.mode ) {
    string s1 = mode.substr(0,5);
    string s2 = obs.mode.substr(0,5);

    if( !(s1==s2 && s1=="2-bit") ){
      if (verbose || combinable_verbose)
	cerr << "dsp::Observation::combinable different mode: '"
	     << mode << "' and '" << obs.mode << "'" << endl;
      can_combine = false;
    }
  }
  
  if (machine != obs.machine) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different machine:"
	   << machine << " and " << obs.machine << endl;
    can_combine = false;
  }
  
  if( fabs(dispersion_measure - obs.dispersion_measure) > eps) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different dispersion measure:"
	   << dispersion_measure << " and " << obs.dispersion_measure << endl;
    can_combine = false;
  }
  
  if( fabs(rotation_measure - obs.rotation_measure) > eps) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different rotation measure:"
	   << rotation_measure << " and " << obs.rotation_measure << endl;
    can_combine = false;
  }

  return can_combine;
}

bool dsp::Observation::contiguous (const Observation & obs, 
                                   bool verbose_on_failure,
                                   int ichan, int ipol) const
{
  if (verbose)
    cerr << "dsp::Observation::contiguous this=" << this << " obs=" << &obs
         << endl;

  double difference = (obs.get_start_time() - get_end_time()).in_seconds();

  if (verbose)
    cerr << "dsp::Observation::contiguous difference=" << difference
         << "s rate=" << rate << "Hz" << endl;

  bool combinable = obs.combinable (*this,false,verbose_on_failure,ichan,ipol);
  bool contiguous = fabs(difference) < 0.9/rate;

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

  if (verbose)
    cerr << "dsp::Observation::contiguous return" << endl;

  return contiguous && combinable;
}

dsp::Observation::Observation (const Observation & in_obs)
  : cerr (in_obs.cerr.rdbuf())
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

  dual_sideband = in_obs.dual_sideband;

  set_centre_frequency   ( in_obs.get_centre_frequency() );
  set_bandwidth          ( in_obs.get_bandwidth() );
  set_dispersion_measure ( in_obs.get_dispersion_measure() );
  set_rotation_measure   ( in_obs.get_rotation_measure() );

  set_start_time  ( in_obs.get_start_time() );

  set_rate        ( in_obs.get_rate() );
  set_scale       ( in_obs.get_scale() );
  set_swap        ( in_obs.get_swap() );
  set_dc_centred  ( in_obs.get_dc_centred() );

  set_identifier  ( in_obs.get_identifier() );
  set_machine     ( in_obs.get_machine() );
  set_mode        ( in_obs.get_mode() );
  set_calfreq     ( in_obs.get_calfreq());

  return *this;
}

// returns the centre_frequency of the ichan channel
double dsp::Observation::get_centre_frequency (unsigned ichan) const
{
  unsigned swap_chan = 0;
  if (swap)
    swap_chan = get_nchan()/2;

  double channel = double ( (ichan+swap_chan) % get_nchan() );

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
void dsp::Observation::change_start_time (int64 samples)
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

