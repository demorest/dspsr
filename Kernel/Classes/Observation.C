#include <stdio.h>
#include <math.h>

#include <string>

#include "dsp/Observation.h"
#include "dsp/Telescope.h"

#include "angle.h"
#include "MJD.h"
#include "Types.h"

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
  between_channel_dm = 0.0;

  domain = "Time";  /* cf 'Fourier' */
  last_ondisk_format = "raw"; /* cf 'CoherentFB' or 'Digi' etc */
}

double dsp::Observation::get_highest_frequency(double max_freq, unsigned chanstart,unsigned chanend){
  chanend = min(nchan,chanend);

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
  chanend = min(nchan,chanend);

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
  return Signal::valid_state(get_state(),get_ndim(),get_npol(),reason);
}
  
bool dsp::Observation::get_detected () const
{
  return (state != Signal::Nyquist && state != Signal::Analytic);
}

/* this returns a flag that is true if the Observations may be combined 
   It doesn't check the start times- you have to do that yourself!
*/
bool dsp::Observation::combinable (const Observation & obs, bool different_bands, bool combinable_verbose, int ichan, int ipol) const
{
  double eps = 0.000001;
  bool can_combine = true;

  if (telescope != obs.telescope) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different telescope:"
	   << telescope << " and " << obs.telescope << endl;
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
    if( fabs(bandwidth/float(nchan) - obs.bandwidth) > eps ) {
      if (verbose || combinable_verbose)
	cerr << "dsp::Observation::combinable different channel bandwidth:"
	     << bandwidth/float(nchan) << " and " << obs.bandwidth << endl;
      can_combine = false;
    }
  }
  else if( fabs(bandwidth-obs.bandwidth) > eps ) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different bandwidth:"
	   << bandwidth << " and " << obs.bandwidth << endl;
    can_combine = false;
  }

  if (nchan != obs.nchan && ichan<0) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different nchan:"
	   << nchan << " and " << obs.nchan << endl;
    can_combine = false;
  }
 
  if (npol != obs.npol && ipol<0 ) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different npol:"
	   << npol << " and " << obs.npol << endl;
    can_combine = false;
  }

  if (ndim != obs.ndim) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different ndim:"
	   << ndim << " and " << obs.ndim << endl;
    can_combine = false;
  }

  if (nbit != obs.nbit) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different nbit:"
	   << nbit << " and " << obs.nbit << endl;
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
  
  if( fabs(rate-obs.rate)/rate > 0.01 ) { /* ie must be within 1% */
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
  
  if (mode != obs.mode) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different mode:"
	   << mode << " and " << obs.mode << endl;
    can_combine = false;
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

  if( different_bands ){
    float this_lo = get_centre_frequency() - fabs(get_bandwidth())/2.0;
    float this_hi = get_centre_frequency() + fabs(get_bandwidth())/2.0;
    float obs_lo = obs.get_centre_frequency() - fabs(obs.get_bandwidth())/2.0;
    float obs_hi = obs.get_centre_frequency() + fabs(obs.get_bandwidth())/2.0;
    
    bool bands_meet = false;
    float eps = 0.000001;

    if( fabs(this_hi-obs_lo)<eps || fabs(this_lo-obs_hi)<eps )
      bands_meet = true;
    
    if( !bands_meet ){
      if( verbose || combinable_verbose )
	fprintf(stderr,"dsp::Observation::combinable bands don't meet- this is centred at %f with bandwidth %f.  obs is centred at %f with bandwidth %f\n",
		get_centre_frequency(), fabs(get_bandwidth()),
		obs.get_centre_frequency(), fabs(obs.get_bandwidth()));
      can_combine = false;
    }
  }
  
  return can_combine;
}

bool dsp::Observation::contiguous (const Observation & obs, bool verbose_on_failure, int ichan, int ipol) const
{
  if( verbose )
    fprintf(stderr,"In dsp::Observation::contiguous() with this=%p and obs=%p\n",
	    this,&obs);

  double difference = fabs((get_end_time() - obs.get_start_time()).in_seconds());

  if( verbose ){
    fprintf(stderr,"Got difference=%f seconds and rate=%f\n",
	    difference,rate);
    fprintf(stderr,"difference=%f   0.9/rate=%f\n",
	    difference,0.9/rate);
  }

  bool combine = obs.combinable(*this,false,verbose_on_failure,ichan,ipol);

  bool ret = ( combine && difference < 0.9/rate );

  if ( !ret && verbose_on_failure ) {
    fprintf(stderr,"dsp::Observation::contiguous() returning false as:\n");
    fprintf(stderr,"combinable(obs)=%d\n",combine);
    fprintf(stderr,"get_start_time().in_seconds()    =%f\n",
	    get_start_time().in_seconds());    
    fprintf(stderr,"get_end_time().in_seconds()      =%f\n",
	    get_end_time().in_seconds());    
    fprintf(stderr,"obs.get_start_time().in_seconds()=%f\n",
	    obs.get_start_time().in_seconds());
    fprintf(stderr,"difference                       =%f\n",fabs(difference));
    fprintf(stderr,"difference needed to be less than %f\n",0.9/rate);    
    fprintf(stderr,"ndat="UI64" and rate=%f.  obs.ndat="UI64" obs.rate=%f\n",
	    ndat,rate,obs.ndat,obs.rate);
  } 

  return ret;
}

void dsp::Observation::set_telescope_code (char _telescope)
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

  set_basis       ( in_obs.get_basis() );
  set_state       ( in_obs.get_state() );
  set_type        ( in_obs.get_type() );

  set_ndim        ( in_obs.get_ndim() );
  set_nchan       ( in_obs.get_nchan() );
  set_npol        ( in_obs.get_npol() );
  set_nbit        ( in_obs.get_nbit() );
  set_ndat        ( in_obs.get_ndat() );

  set_telescope_code   ( in_obs.get_telescope_code() );
  set_source      ( in_obs.get_source() );
  set_coordinates ( in_obs.get_coordinates() );

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

  set_domain( in_obs.get_domain() );
  set_last_ondisk_format( in_obs.get_last_ondisk_format() );

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
    swap_chan = nchan/2;

  double channel = double ( (ichan+swap_chan) % nchan );

  return get_base_frequency() + channel * bandwidth / double(nchan);
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
    return centre_frequency - 0.5*bandwidth + 0.5*bandwidth/double(nchan);
}

void dsp::Observation::get_minmax_frequencies (double& min, double& max) const
{
  min = get_base_frequency();
  max = min + bandwidth*(1.0-1.0/double(nchan));

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
    ndat /= 2;         // number of complex samples
    rate /= 2.0;       // samples are now complex at half the rate
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

//! Returns all information contained in this class into the string info_string
bool dsp::Observation::obs2string(string& ss) const{
  string ui64(UI64);
  ui64.replace(0,1,"%16");

  char dummy[256];
  char loony[256];

  sprintf(dummy,"dsp::Observation_data\n"); ss+= dummy;
  sprintf(loony,"NDAT\t%s\n",ui64.c_str()); sprintf(dummy,loony,ndat); ss += dummy;
  sprintf(dummy,"TELESCOPE\t%c\n",telescope); ss += dummy;
  sprintf(dummy,"SOURCE\t%s\n",source.c_str()); ss += dummy;
  sprintf(dummy,"CENTRE_FREQUENCY\t%.16f\n",centre_frequency); ss += dummy;
  sprintf(dummy,"BANDWIDTH\t%.16f\n",bandwidth); ss += dummy;
  sprintf(dummy,"NCHAN\t%d\n",nchan); ss += dummy;
  sprintf(dummy,"NPOL\t%d\n",npol); ss += dummy;
  sprintf(dummy,"NDIM\t%d\n",ndim); ss += dummy;
  sprintf(dummy,"NBIT\t%d\n",nbit); ss += dummy;
  sprintf(dummy,"TYPE\t%s\n",Signal::Source2string(type).c_str()); ss += dummy;
  sprintf(dummy,"STATE\t%s\n",Signal::State2string(state).c_str()); ss += dummy;
  sprintf(dummy,"BASIS\t%s\n",Signal::Basis2string(basis).c_str()); ss += dummy;
  sprintf(dummy,"RATE\t%.16f\n",rate); ss += dummy;
  sprintf(dummy,"START_TIME\t%s\n",start_time.printdays(15).c_str()); ss += dummy;
  sprintf(dummy,"SCALE\t%.16f\n",scale); ss += dummy;
  sprintf(dummy,"SWAP\t%s\n",swap?"true":"false"); ss += dummy;
  sprintf(dummy,"DC_CENTRED\t%s\n",dc_centred?"true":"false"); ss += dummy;
  sprintf(dummy,"IDENTIFIER\t%s\n",identifier.c_str()); ss += dummy;
  sprintf(dummy,"MODE\t%s\n",mode.c_str()); ss += dummy;
  sprintf(dummy,"MACHINE\t%s\n",machine.c_str()); ss += dummy;

  /* COORDINATES is stored as RAJ and DECJ */
  sprintf(dummy,"RAJ\t%s\n",coordinates.ra().getHMS().c_str()); ss += dummy;
  sprintf(dummy,"DECJ\t%s\n",coordinates.dec().getDMS().c_str()); ss += dummy;
  
  sprintf(dummy,"DISPERSION_MEASURE\t%.16f\n",dispersion_measure); ss += dummy;
  sprintf(dummy,"BETWEEN_CHANNEL_DM\t%.16f\n",between_channel_dm); ss += dummy;
  sprintf(dummy,"DOMAIN\t%s\n",domain.c_str()); ss += dummy;
  sprintf(dummy,"LAST_ONDISK_FORMAT\t%s\n",last_ondisk_format.c_str()); ss += dummy;

  return true;
}
    
//! Writes all information contained in this class into the fptr at the current file offset
bool dsp::Observation::obs2file(FILE* fptr){
  string ss;
  if( !obs2string(ss) ){
    fprintf(stderr,"dsp::Observation::retrieve() failed to write to fptr because string version failed\n");
    fclose(fptr);
    return false;
  }

  fprintf(fptr,"%s",ss.c_str());

  return true;
}

//! The file pointer must be appropriately seeked
bool dsp::Observation::file2obs(FILE* fptr){
  if( verbose )
    fprintf(stderr,"In dsp::Observation::file2obs()\n");

  if( !fptr ){
    cerr << "dsp::Observation::file2obs() returning false as fptr=NULL\n";
    return false;
  }

  char dummy[1024];
  char moron[1024];

  int scanned = fscanf(fptr,"%s\n",dummy);
  if( scanned!=1 ){
    cerr << "dsp::Observation::file2obs() returning false as could not read first line\n";
    return false;
  }

  if( string(dummy)!=string("dsp::Observation_data") ){
    cerr << "dsp::Observation::file2obs() returning false as first line is not 'dsp::Observation_data'.  It is '" << string(dummy) << "'\n";
    return false;
  }

  string ui64(UI64);
  ui64.replace(0,1,"%16");
  sprintf(moron,"NDAT\t%s\n",ui64.c_str());

  int ret = fscanf(fptr,moron,&ndat); if(verbose) fprintf(stderr,"Got ndat="UI64"\n",ndat); 
  if( ret!=1 )
    throw Error(FailedCall,"dsp::Observation::file2obs()",
		"Failed to fscanf ndat\n");

  fscanf(fptr,"TELESCOPE\t%c\n",&telescope);  if(verbose) fprintf(stderr,"Got telescope=%c\n",telescope); 
  retrieve_cstring(fptr,"SOURCE\t",dummy); source = dummy;  if(verbose) fprintf(stderr,"Got source=%s\n",source.c_str()); 
  fscanf(fptr,"CENTRE_FREQUENCY\t%lf\n",&centre_frequency);  if(verbose) fprintf(stderr,"Got centre_frequency=%f\n",centre_frequency); 
  fscanf(fptr,"BANDWIDTH\t%lf\n",&bandwidth);  if(verbose) fprintf(stderr,"Got bandwidth=%f\n",bandwidth); 
  fscanf(fptr,"NCHAN\t%d\n",&nchan);  if(verbose) fprintf(stderr,"Got nchan=%d\n",nchan); 
  fscanf(fptr,"NPOL\t%d\n",&npol);  if(verbose) fprintf(stderr,"Got npol=%d\n",npol); 
  fscanf(fptr,"NDIM\t%d\n",&ndim);  if(verbose) fprintf(stderr,"Got ndim=%d\n",ndim); 
  fscanf(fptr,"NBIT\t%d\n",&nbit);  if(verbose) fprintf(stderr,"Got nbit=%d\n",nbit); 
  retrieve_cstring(fptr,"TYPE\t",dummy); type = Signal::string2Source(dummy);  if(verbose) fprintf(stderr,"Got type=%s\n",Signal::Source2string(type).c_str()); 
  retrieve_cstring(fptr,"STATE\t",dummy); state = Signal::string2State(dummy);  if(verbose) fprintf(stderr,"Got state=%s\n",Signal::State2string(state).c_str()); 
  retrieve_cstring(fptr,"BASIS\t",dummy); basis = Signal::string2Basis(dummy);  if(verbose) fprintf(stderr,"Got basis=%s\n",Signal::Basis2string(basis).c_str()); 
  fscanf(fptr,"RATE\t%lf\n",&rate);  if(verbose) fprintf(stderr,"Got rate=%f\n",rate); 
  retrieve_cstring(fptr,"START_TIME\t",dummy); start_time = MJD(dummy);  if(verbose) fprintf(stderr,"Got start_time=%s\n",start_time.printdays(15).c_str()); 
  fscanf(fptr,"SCALE\t%lf\n",&scale);  if(verbose) fprintf(stderr,"Got scale=%f\n",scale); 
  fscanf(fptr,"SWAP\t%s\n",dummy);
  if( string(dummy)==string("true") )
    swap = true;
  else
    swap = false; 
  if(verbose) fprintf(stderr,"Got swap=%d\n",swap); 
  fscanf(fptr,"DC_CENTRED\t%s\n",dummy);
  if( string(dummy)==string("true") )
    dc_centred = true;
  else
    dc_centred = false;
  if(verbose) fprintf(stderr,"Got dc_centred=%d\n",dc_centred); 
  retrieve_cstring(fptr,"IDENTIFIER\t",dummy); identifier=dummy;  if(verbose) fprintf(stderr,"Got identifier=%s\n",identifier.c_str()); 
  retrieve_cstring(fptr,"MODE\t",dummy); mode = dummy;  if(verbose) fprintf(stderr,"Got mode=%s\n",mode.c_str()); 
  retrieve_cstring(fptr,"MACHINE\t",dummy); machine = dummy; if(verbose) fprintf(stderr,"Got machine=%s\n",machine.c_str()); 

  /* COORDINATES is stored as RAJ and DECJ */
  retrieve_cstring(fptr,"RAJ\t",dummy); 
  retrieve_cstring(fptr,"DECJ\t",moron);
  coordinates.setHMSDMS(dummy,moron);
  if(verbose) fprintf(stderr,"Got coordinates=%s\n",coordinates.getHMSDMS().c_str());   

  fscanf(fptr,"DISPERSION_MEASURE\t%lf\n",&dispersion_measure); if(verbose) fprintf(stderr,"Got dispersion_measure=%f\n",dispersion_measure); 
  fscanf(fptr,"BETWEEN_CHANNEL_DM\t%lf\n",&between_channel_dm); if(verbose) fprintf(stderr,"Got between_channel_dm=%f\n",between_channel_dm); 
  retrieve_cstring(fptr,"DOMAIN\t",dummy); domain = dummy; if(verbose) fprintf(stderr,"Got domain=%s\n",domain.c_str());
  retrieve_cstring(fptr,"LAST_ONDISK_FORMAT\t",dummy); last_ondisk_format = dummy; if(verbose) fprintf(stderr,"Got last_ondisk_format=%s\n",last_ondisk_format.c_str());


  string ss;
  if( !obs2string(ss) )
    throw Error(InvalidState,"dsp::Observation::file2string()",
		"Couldn't obs2string!\n");

  if( verbose )
    fprintf(stderr,"Data got in:\n%s\n",ss.c_str());

  return true;
}

