/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <stdio.h>
#include <math.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

#include <string>
#include <vector>

#include "Angle.h"
#include "MJD.h"
#include "Types.h"
#include "dirutil.h"
#include "Error.h"
#include "tempo++.h"

#include "environ.h"

#include "dsp/dspExtension.h"
#include "dsp/Observation.h"

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
  Observation::set_ndat( 0 );
  Observation::set_nchan( 1 );
  Observation::set_npol( 1 );
  Observation::set_ndim( 1 );
  Observation::set_nbit( 0 );
  Observation::set_calfreq(0.0);

  type = Signal::Pulsar;
  state = Signal::Intensity;
  basis = Signal::Linear;

  telescope = "unknown";
  receiver = "unknown";
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
  between_channel_dm = 0.0;

  dual_sideband = -1;

  domain = "Time";  /* cf 'Fourier' */
  last_ondisk_format = "raw"; /* cf 'CoherentFB' or 'Digi' etc */
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
     
double dsp::Observation::get_highest_frequency(double max_freq, unsigned chanstart,unsigned chanend){
  chanend = min(get_nchan(),chanend);

  double sign_change = 1;

  if( get_centre_frequency()<0.0 )
    sign_change = -1;

  unsigned j=chanstart;
  while( j<chanend && sign_change*get_centre_frequency(j)>max_freq ){
    //fprintf(stderr,"dsp::Observation::get_highest_frequency() Chan %d at %f is higher than %f\n",
    //    j,sign_change*get_centre_frequency(j),max_freq);
    j++;
  }

  if( j==chanend )
    throw Error(InvalidParam,"dsp::Observation::get_highest_frequency()",
		"Your max_freq of %f is higher than all the centre frequencies available",
		max_freq);

  double highest_frequency = sign_change*get_centre_frequency(j);

  for( unsigned i=j; i<chanend; i++){
    if( sign_change*get_centre_frequency(i)>max_freq ){
      //      fprintf(stderr,"dsp::Observation::get_highest_frequency() Chan %d at %f is higher than %f\n",
      //    i,sign_change*get_centre_frequency(i),max_freq);
    }
    else{
      highest_frequency = max(highest_frequency,sign_change*get_centre_frequency(i));
    }
  }

  return highest_frequency;
}

double dsp::Observation::get_lowest_frequency(double min_freq, unsigned chanstart,unsigned chanend){
  chanend = min(get_nchan(),chanend);

  double sign_change = 1;

  if( get_centre_frequency()<0.0 )
    sign_change = -1;

  unsigned j=chanstart;
  while( j<chanend && sign_change*get_centre_frequency(j)<min_freq ){
    //    fprintf(stderr,"dsp::Observation::get_lowest_frequency() Chan %d at %f is lower than %f\n",
    //    j,sign_change*get_centre_frequency(j),min_freq);
    j++;
  }

  if( j==chanend )
    throw Error(InvalidParam,"dsp::Observation::get_lowest_frequency()",
		"Your min_freq of %f is lower than all the centre frequencies available",
		min_freq);

  double lowest_frequency = sign_change*get_centre_frequency(0);

  for( unsigned i=j; i<chanend; i++){
    if( sign_change*get_centre_frequency(i)<min_freq ){
      //fprintf(stderr,"dsp::Observation::get_lowest_frequency() Chan %d at %f is lower than %f\n",
      //    i,sign_change*get_centre_frequency(i),min_freq);
    }
    else{
      lowest_frequency = min(lowest_frequency,sign_change*get_centre_frequency(i));
    }
  }

  return lowest_frequency;
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
  bool can_combine = ordinary_checks( obs, different_bands, combinable_verbose, ichan, ipol );

  if( different_bands && !bands_adjoining(obs) ){
    if( verbose || combinable_verbose )
	fprintf(stderr,"dsp::Observation::combinable bands don't meet- this is centred at %f with bandwidth %f.  obs is centred at %f with bandwidth %f\n",
		get_centre_frequency(), fabs(get_bandwidth()),
		obs.get_centre_frequency(), fabs(obs.get_bandwidth()));
      can_combine = false;
  }

  return can_combine;
}

bool dsp::Observation::ordinary_checks(const Observation & obs, bool different_bands, bool combinable_verbose, int ichan, int ipol) const {
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

  if (source != obs.source) {
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
  
  if (!combinable_rate (obs.rate)) {
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
  
  if( domain != obs.domain ) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different domains:"
	   << domain << " and " << obs.domain << endl;
    can_combine = false;
  }
  
  if( fabs(dispersion_measure - obs.dispersion_measure) > eps) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different dispersion measure:"
	   << dispersion_measure << " and " << obs.dispersion_measure << endl;
    can_combine = false;
  }
  
  if( fabs(between_channel_dm - obs.between_channel_dm) > eps) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different dispersion measure:"
	   << between_channel_dm << " and " << obs.between_channel_dm << endl;
    can_combine = false;
  }

  return can_combine;
}

bool dsp::Observation::bands_adjoining(const Observation& obs) const {
  float this_lo = get_centre_frequency() - fabs(get_bandwidth())/2.0;
  float this_hi = get_centre_frequency() + fabs(get_bandwidth())/2.0;
  float obs_lo = obs.get_centre_frequency() - fabs(obs.get_bandwidth())/2.0;
  float obs_hi = obs.get_centre_frequency() + fabs(obs.get_bandwidth())/2.0;
  
  float eps = 0.000001;
  
  if( fabs(this_hi-obs_lo)<eps || fabs(this_lo-obs_hi)<eps )
    return true;
  
  if( verbose )
    fprintf(stderr,"dsp::Observation::bands_adjoining) returning false\n");

  return false;
}

bool dsp::Observation::bands_combinable(const Observation& obs,bool combinable_verbose) const{
  return ordinary_checks(obs,true,combinable_verbose);
}

/* return true if the test_rate is within 1% of the rate attribute */
bool dsp::Observation::combinable_rate (double test_rate) const
{
  return fabs(rate-test_rate)/rate < 0.01;
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

  set_centre_frequency ( in_obs.get_centre_frequency() );
  set_bandwidth   ( in_obs.get_bandwidth() );
  set_dispersion_measure   ( in_obs.get_dispersion_measure() );
  set_between_channel_dm( in_obs.get_between_channel_dm() );

  set_start_time  ( in_obs.get_start_time() );

  set_rate        ( in_obs.get_rate() );
  set_scale       ( in_obs.get_scale() );
  set_swap        ( in_obs.get_swap() );
  set_dc_centred  ( in_obs.get_dc_centred() );

  set_identifier  ( in_obs.get_identifier() );
  set_machine     ( in_obs.get_machine() );
  set_mode        ( in_obs.get_mode() );
  set_calfreq     ( in_obs.get_calfreq());

  set_domain( in_obs.get_domain() );
  set_last_ondisk_format( in_obs.get_last_ondisk_format() );

  extensions.resize( 0 );

  for( unsigned iext=0; iext<in_obs.get_nextensions(); iext++)
    add( in_obs.get_extension(iext)->clone() );

  return *this;
}

dsp::Observation& dsp::Observation::swap_data(Observation& obs){
  dsp::Observation temp = *this;
  operator=( obs );
  obs = temp;

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

//! Returns the centre frequency of the ichan'th frequency ordered channel in MHz.
double dsp::Observation::get_ordered_cfreq(unsigned ichan){
  return get_centre_frequency(ichan);
}

// returns the centre_frequency of the first channel
double dsp::Observation::get_base_frequency () const
{
  if (dc_centred)
    return centre_frequency - 0.5*bandwidth;
  else
    return centre_frequency - 0.5*bandwidth + 0.5*bandwidth/double(get_nchan());
}

void dsp::Observation::get_minmax_frequencies (double& min, double& max) const
{
  min = get_base_frequency();
  max = min + bandwidth*(1.0-1.0/double(get_nchan()));

  if (min > max)
    std::swap (min, max);
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

//! Constructs the CPSR2-header parameter, OFFSET
uint64 dsp::Observation::get_offset(){
  MJD obs_start(identifier);
  double time_offset = (get_start_time()-obs_start).in_seconds();

  return get_nbytes(uint64(time_offset*rate));
}

//! Adds a dspExtension
void dsp::Observation::add(dspExtension* extension){
  if( extension->must_only_have_one() )
    for( unsigned i=0; i<extensions.size(); i++)
      if( extensions[i]->get_name()==extension->get_name() )
	throw Error(InvalidState,"dsp::Observation::add()",
		    "You can only have one '%s' dspExtension, but you are trying to add your second!",
		    extension->get_name().c_str());

  extensions.push_back( extension );
}

//! Removes a dspExtension
Reference::To<dsp::dspExtension> 
dsp::Observation::remove_extension(const string& ext_name){
  for( unsigned i=0; i<extensions.size(); i++){
    if( extensions[i]->get_name() == ext_name ){
      Reference::To<dspExtension> removed = extensions[i];
      extensions.erase( extensions.begin()+i );
      return removed;
    }
  }
  return 0;
}

//! Returns true if one of the stored dspExtensions has this name
bool dsp::Observation::has(const string& extension_name){
  for( unsigned i=0; i<extensions.size(); i++)
    if( extensions[i]->get_name()==extension_name )
      return true;

  return false;
}

//! Returns the number of dspExtensions currently stored
unsigned dsp::Observation::get_nextensions() const {
  return extensions.size();
}

//! Returns the i'th dspExtension stored
dsp::dspExtension* dsp::Observation::get_extension(unsigned iext){
  if( iext >= extensions.size() )
    throw Error(InvalidParam,"dsp::Observation::get_extension()",
		"You requested extension '%d' but there are only %d extensions stored",iext,extensions.size());
  return extensions[iext].get();
}

//! Returns the i'th dspExtension stored
const dsp::dspExtension* dsp::Observation::get_extension(unsigned iext) const{
  if( iext >= extensions.size() )
    throw Error(InvalidParam,"dsp::Observation::get_extension()",
		"You requested extension '%d' but there are only %d extensions stored",iext,extensions.size());
  return extensions[iext].get();
}

//! Return the end time of the trailing edge of the last time sample
// Returns correct answer if ndat=rate=0 and avoids division by zero
MJD dsp::Observation::get_end_time () const
{
  if( ndat==0 )
    return start_time;
  return start_time + double(ndat) / rate;
}

