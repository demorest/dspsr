#include <stdio.h>
#include <math.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

#include <string>

#include "dsp/Observation.h"
#include "Telescope.h"

#include "angle.h"
#include "MJD.h"
#include "Types.h"
#include "Header.h"
#include "genutil.h"
#include "dirutil.h"
#include "string_utils.h"
#include "Error.h"
#include "Reference.h"

bool dsp::Observation::verbose = false;

dsp::Observation::Observation ()
{
  init ();
}

void dsp::Observation::init ()
{
  set_ndat( 0 );
  nchan = 1;
  npol = 1;
  set_ndim( 1 );
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
    set_ndim(1);
  else if (state == Signal::Analytic)
    set_ndim(2);
  else if (state == Signal::Intensity){
    set_ndim(1);
    npol = 1;
  }
  else if (state == Signal::PPQQ){
    set_ndim(1);
    npol = 2;
  }
  else if (state == Signal::Coherence){
    /* best not to muck with kludges */
  }
  else if (state == Signal::Stokes){
    /* best not to muck with kludges */
  }
  else if (state == Signal::Invariant){
    set_ndim(1);
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

  if (get_ndim() != obs.get_ndim()) {
    if (verbose || combinable_verbose)
      cerr << "dsp::Observation::combinable different ndim:"
	   << get_ndim() << " and " << obs.get_ndim() << endl;
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
    fprintf(stderr,"get_start_time().in_seconds()    =%.9f\n",
	    get_start_time().in_seconds());    
    fprintf(stderr,"get_end_time().in_seconds()      =%.9f\n",
	    get_end_time().in_seconds());    
    fprintf(stderr,"obs.get_start_time().in_seconds()=%.9f\n",
	    obs.get_start_time().in_seconds());
    fprintf(stderr,"difference                       =%.9f\n",fabs(difference));
    fprintf(stderr,"difference needed to be less than %.9f\n",0.9/rate);    
    fprintf(stderr,"ndat="UI64" and rate=%f.  obs.ndat="UI64" obs.rate=%f\n",
	    get_ndat(),rate,obs.get_ndat(),obs.rate);
  } 

  if( verbose )
    fprintf(stderr,"Returning from dsp::Observation::contiguous() with %d\n",
	    ret);

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
    // if (centre_frequency > 2000.0)
    //  basis = Signal::Circular;
    // with the advent of the 1050 the above assumption is no longer valid

  }
  else if (telescope == Telescope::ATCA)
    basis = Signal::Circular;
  else if (telescope == Telescope::Tidbinbilla)
    basis = Signal::Circular;
  else if (telescope == Telescope::Arecibo)
    basis = Signal::Circular;
  else if (telescope == Telescope::Hobart)
    basis = Signal::Circular;
  else if (telescope == Telescope::GreenBank) {
    fprintf(stderr,"WARNING Assuming GBT is Circular\n");
    basis = Signal::Circular;
  }
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

//! Returns all information contained in this class into the return value
string dsp::Observation::obs2string() const {
  Reference::To<Header> header(obs2Header());
  return header->write_string();
}

//! Returns a Header that stores the info in the class
Reference::To<Header> dsp::Observation::obs2Header(Header* hdr) const{
  Reference::To<Header> h(hdr);
  if( !h.ptr() ){
    h = new Header;
    h->set_size( 4096 );
    h->set_id("dsp::Observation_data");
    h->set_version( get_version() );
  }

  h->add_token("NDAT",get_ndat());
  h->add_token("TELESCOPE",telescope);
  h->add_token("SOURCE",source);
  h->add_token("CENTRE_FREQUENCY",centre_frequency);
  h->add_token("BANDWIDTH",bandwidth);
  h->add_token("NCHAN",nchan);
  h->add_token("NPOL",npol);
  h->add_token("NDIM",get_ndim());
  h->add_token("NBIT",nbit);
  h->add_token("TYPE",Signal::Source2string(type));
  h->add_token("STATE",Signal::State2string(state));
  h->add_token("BASIS",Signal::Basis2string(basis));
  h->add_token("RATE",rate);
  h->add_token("START_TIME",start_time.printdays(15));
  h->add_token("SCALE",scale);
  h->add_token("SWAP",swap?"true":"false");
  h->add_token("DC_CENTRED",dc_centred?"true":"false");
  h->add_token("IDENTIFIER",identifier);
  h->add_token("MODE",mode);
  h->add_token("MACHINE",machine);

  // COORDINATES is stored as RAJ and DECJ
  h->add_token("RAJ",coordinates.ra().getHMS());
  h->add_token("DECJ",coordinates.dec().getDMS());

  h->add_token("DISPERSION_MEASURE",dispersion_measure);
  h->add_token("BETWEEN_CHANNEL_DM",between_channel_dm);
  h->add_token("DOMAIN",domain);
  h->add_token("LAST_ONDISK_FORMAT",last_ondisk_format);

  return h;
}

  /*
//! Returns all information contained in this class into the string info_string
bool dsp::Observation::obs2string(string& info_string) const{
  string ui64(UI64);
  ui64.replace(0,1,"%16");

  char dummy[256];
  char loony[256];

  sprintf(dummy,"dsp::Observation_data\n"); ss+= dummy;
  sprintf(loony,"NDAT\t%s\n",ui64.c_str()); sprintf(dummy,loony,get_ndat()); ss += dummy;
  sprintf(dummy,"TELESCOPE\t%c\n",telescope); ss += dummy;
  sprintf(dummy,"SOURCE\t%s\n",source.c_str()); ss += dummy;
  sprintf(dummy,"CENTRE_FREQUENCY\t%.16f\n",centre_frequency); ss += dummy;
  sprintf(dummy,"BANDWIDTH\t%.16f\n",bandwidth); ss += dummy;
  sprintf(dummy,"NCHAN\t%d\n",nchan); ss += dummy;
  sprintf(dummy,"NPOL\t%d\n",npol); ss += dummy;
  sprintf(dummy,"NDIM\t%d\n",get_ndim()); ss += dummy;
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

  // COORDINATES is stored as RAJ and DECJ
  sprintf(dummy,"RAJ\t%s\n",coordinates.ra().getHMS().c_str()); ss += dummy;
  sprintf(dummy,"DECJ\t%s\n",coordinates.dec().getDMS().c_str()); ss += dummy;
  
  sprintf(dummy,"DISPERSION_MEASURE\t%.16f\n",dispersion_measure); ss += dummy;
  sprintf(dummy,"BETWEEN_CHANNEL_DM\t%.16f\n",between_channel_dm); ss += dummy;
  sprintf(dummy,"DOMAIN\t%s\n",domain.c_str()); ss += dummy;
  sprintf(dummy,"LAST_ONDISK_FORMAT\t%s\n",last_ondisk_format.c_str()); ss += dummy;

  return true;
}
  */

//! Writes all information contained in this class into the specified filename
void dsp::Observation::obs2file(string filename, int64 offset) const{
  int fd = ::open(filename.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);

  obs2file(fd,offset, SEEK_SET);

  ::close(fd);
}

//! Writes all information contained in this class into the file descriptor
void dsp::Observation::obs2file(int fd, int64 offset, int whence) const{
  Reference::To<Header> header(obs2Header());

  header->write(fd,offset,whence);
}

//! Opposite of obs2file
void dsp::Observation::file2obs(string filename, int64 offset){
  if( !file_exists(filename.c_str()) )
    throw Error(InvalidParam,"dsp::Observation::file2obs()",
		"File '%s' does not exist",filename.c_str());

  int fd = ::open(filename.c_str(), O_RDONLY);

  file2obs(fd,offset,SEEK_SET);

  ::close(fd);
}

//! Worker function for file2obs that seeks through to correct place in file
void dsp::Observation::file_seek(int fd,int64 offset,int whence){
  if( fd<0 )
    throw Error(InvalidParam,"dsp::Observation::file_seek()",
                "Invalid file descriptor passed in");

  int64 seeked = ::lseek( fd, offset, whence );

  if( whence==SEEK_SET && seeked != offset )
    throw Error(FailedCall,"dsp::Observation::file_seek()",
                "Return value from seek was '"UI64"'.  cf requested seek="UI64,
		seeked, offset);
}

//! Determines the version of the dsp::Observation printout
float dsp::Observation::get_version(int fd){
  string firstline = read_line(fd);

  vector<string> words = stringdecimate(firstline," \t");

  if( words.size() != 3 ) // Pre-Header days of dsp::Observation
    return 1.0;  
  
  float version = Header::parse_version(firstline);

  // Rewind the file descriptor by the first line and the newline character
  ::lseek(fd,-int64(firstline.size()+1),SEEK_CUR);

  return version;
}

//! Opposite of obs2file
void dsp::Observation::file2obs(int fd, int64 offset, int whence){
  file_seek( fd, offset, whence);

  if( get_version(fd) < 2.0 ){
    old_file2obs(fd);
    return;
  }    

  Reference::To<Header> hdr(new Header(fd) );

  Header2obs(hdr);
}

//! Initializes the Observation from a parsed-Header
void dsp::Observation::Header2obs(Reference::To<Header> h){
  set_ndat( h->retrieve_token<uint64>("NDAT") );
  telescope = h->retrieve_token<char>("TELESCOPE");
  source = h->retrieve_token<string>("SOURCE");
  centre_frequency = h->retrieve_token<double>("CENTRE_FREQUENCY");
  bandwidth = h->retrieve_token<double>("BANDWIDTH");
  nchan = h->retrieve_token<unsigned>("NCHAN");
  npol = h->retrieve_token<unsigned>("NPOL");
  set_ndim( h->retrieve_token<unsigned>("NDIM") );
  nbit = h->retrieve_token<unsigned>("NBIT");
  type = Signal::string2Source( h->retrieve_token<string>("TYPE") );
  state = Signal::string2State( h->retrieve_token<string>("STATE") );
  basis = Signal::string2Basis( h->retrieve_token<string>("BASIS") );
  rate = h->retrieve_token<double>("RATE") ;
  start_time = MJD( h->retrieve_token<string>("START_TIME") );
  scale = h->retrieve_token<double>("SCALE");
  swap = h->retrieve_token<string>("SWAP")=="true" ? true : false;
  dc_centred = h->retrieve_token<string>("DC_CENTRED")=="true" ? true : false;
  identifier = h->retrieve_token<string>("IDENTIFIER");
  mode = h->retrieve_token<string>("MODE");
  machine = h->retrieve_token<string>("MACHINE");

  // COORDINATES are stored as RAJ and DECJ
  string raj = h->retrieve_token<string>("RAJ");
  string decj = h->retrieve_token<string>("DECJ");
  coordinates.setHMSDMS(raj.c_str(),decj.c_str());

  dispersion_measure = h->retrieve_token<double>("DISPERSION_MEASURE");
  between_channel_dm = h->retrieve_token<double>("BETWEEN_CHANNEL_DM");
  domain = h->retrieve_token<string>("DOMAIN");
  last_ondisk_format = h->retrieve_token<string>("LAST_ONDISK_FORMAT");
}

//! Old pre-Header version of file2obs()
void dsp::Observation::old_file2obs(int fd){
  if( verbose )
    fprintf(stderr,"In dsp::Observation::file2obs()\n");

  FILE* fptr = fdopen( fd, "r");

  if( !fptr )
    throw Error(FailedCall,"dsp::Observation::old_file2obs()",
		"Could not derive a valid file pointer");

  char dummy[1024];
  char moron[1024];

  int scanned = fscanf(fptr,"%s\n",dummy);
  if( scanned!=1 )
    throw Error(FailedCall,"dsp::Observation::old_file2obs()",
		"could not read first line");

  if( string(dummy)!=string("dsp::Observation_data") )
    throw Error(InvalidState,"dsp::Observation::old_file2obs()",
		"first line is not 'dsp::Observation_data'.  It is '%s'",dummy);

  string ui64(UI64);
  ui64.replace(0,1,"%16");
  sprintf(moron,"NDAT\t%s\n",ui64.c_str());

  uint64 dumbo = 0;
  unsigned retardo = 0;

  int ret = fscanf(fptr,moron,&dumbo); set_ndat(dumbo); if(verbose) fprintf(stderr,"Got ndat="UI64"\n",get_ndat()); 
  if( ret!=1 )
    throw Error(FailedCall,"dsp::Observation::old_file2obs()",
		"Failed to fscanf ndat");

  fscanf(fptr,"TELESCOPE\t%c\n",&telescope);  if(verbose) fprintf(stderr,"Got telescope=%c\n",telescope); 
  retrieve_cstring(fptr,"SOURCE\t",dummy); source = dummy;  if(verbose) fprintf(stderr,"Got source=%s\n",source.c_str()); 
  fscanf(fptr,"CENTRE_FREQUENCY\t%lf\n",&centre_frequency);  if(verbose) fprintf(stderr,"Got centre_frequency=%f\n",centre_frequency); 
  fscanf(fptr,"BANDWIDTH\t%lf\n",&bandwidth);  if(verbose) fprintf(stderr,"Got bandwidth=%f\n",bandwidth); 
  fscanf(fptr,"NCHAN\t%d\n",&nchan);  if(verbose) fprintf(stderr,"Got nchan=%d\n",nchan); 
  fscanf(fptr,"NPOL\t%d\n",&npol);  if(verbose) fprintf(stderr,"Got npol=%d\n",npol); 
  fscanf(fptr,"NDIM\t%d\n",&retardo); set_ndim(retardo); if(verbose) fprintf(stderr,"Got ndim=%d\n",get_ndim()); 
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

  if( verbose )
    fprintf(stderr,"Data got in:\n%s\n",obs2string().c_str());

}










