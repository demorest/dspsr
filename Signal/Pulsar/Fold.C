#include <string>

#include <assert.h>

#include "dsp/WeightedTimeSeries.h"

#include "Error.h"

#include "MatchingEphemeris.h"
#include "MatchingPolyco.h"
#include "tempo++.h"
#include "genutil.h"
#include "string_utils.h"

#include "dsp/Fold.h"

dsp::Fold::Fold () : Transformation <const TimeSeries, PhaseSeries> ("Fold", outofplace) 
{
  folding_period = 0;

  requested_nbin = 0;
  folding_nbin = 0;

  reference_phase = 0.0;

  ncoef = 15;
  nspan = 120;

  dispersion_measure = -1.0;

  if (psrdisp_compatible) {
    cerr << "dsp::Fold psrdisp compatibility\n"
      "   using nspan of 960 instead of 120" << endl;
    nspan = 960;
  }

  built = false;

  idat_start = ndat_fold = 0;
}

dsp::Fold::~Fold () { }

//! Prepare for folding the input TimeSeries
void dsp::Fold::prepare ()
{
  if (!input)
    throw Error (InvalidState, "dsp::Fold::prepare", "no input");

  prepare (input);
}

//! Prepare for folding the given Observation
void dsp::Fold::prepare (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::Fold::prepare" << endl;

  string jpulsar = observation->get_source();
  
  if (jpulsar.length() == 0)
    throw Error (InvalidParam, "dsp::Fold::prepare",
		 "empty Observation::source");

  if( folding_period > 0.0 && (folding_period_source==jpulsar || folding_period_source==string()) ){
    fprintf(stderr,"Already got folding_period so returning straight away from dsp::Fold::prepare()\n");
    pulsar_ephemeris = 0;
    folding_polyco = 0;
    built = true;
    return;
  }

  if (jpulsar[0] != 'J')
    jpulsar = "J" + jpulsar;

  if (verbose)
    cerr << "dsp::Fold::prepare source=" << jpulsar << endl;

  // Preference 1: correct polyco already generated
  folding_polyco = choose_polyco (observation->get_start_time(), jpulsar);

  if (folding_polyco)
    return;

  // Preference 2a: Correct ephemeris already generated
  pulsar_ephemeris = choose_ephemeris (jpulsar);

  fprintf(stderr,"dsp::Fold::prepare ephemeris already generated\n");

  // Preference 2b: Generate correct ephemeris
  if (!pulsar_ephemeris){
    fprintf(stderr,"dsp::Fold::prepare ephemeris generating ephemeris\n");
    Reference::To<MatchingEphemeris> mephem = new MatchingEphemeris(jpulsar.c_str(), 0);
    add_pulsar_ephemeris( mephem );
    pulsar_ephemeris = mephem;
  }

  fprintf(stderr,"dsp::Fold::prepare got pulsar_ephemeris->get_psrname='%s'\n",
	  pulsar_ephemeris->psrname().c_str());

#if 0

  dm = ephemeris.get_dm();
  if (verbose)
    cerr << "dsp::Fold::prepare psrephem dm = " << dm << endl;

  TODO - ephemeris and coordinates in archive

  // set the source position
  raw.coordinates.setRadians (ephemeris.jra(), ephemeris.jdec());

  // No longer using the catalogue!!
  if (psrstat != NULL) {
    if (verbose)
      cerr << "Looking up " << info.source << " in catalogue\n";

    // look up the pulsar in the catalogue
    if (!creadcat (jpulsar.c_str(), psrstat))  {
      cerr << "dsp::Fold::prepare error creadcat (" 
	   << jpulsar << ")\n";
      return -1;
    }

    nspan = (*psrstat)->nspan;
    ncoef = (*psrstat)->ncoef;
  }

#endif

  if (verbose)
    cerr << "dsp::Fold::prepare creating polyco" << endl;

  // Preference 2: Generate correct polyco from ephemeris and add it to the list of polycos to choose from
  folding_polyco = get_folding_polyco(pulsar_ephemeris,observation);

  fprintf(stderr,"dsp::Fold got folding_polyco with name '%s'\n",
	  folding_polyco->get_psrname().c_str());
  //exit(0);

#if 0
  doppler = 1.0 + psr_poly->doppler_shift(raw.start_time);
  
  if (verbose)
    cerr << "dsp::Fold::prepare Doppler shift from polyco:" << doppler << endl;

#endif

  // The Fold::prepare method may be called with an Observation state
  // not equal to the state of the input to be folded.  Therefore, the
  // choice of folding_nbin is postponed until the first call to
  // Fold::transformation.

  folding_nbin = 0;

  built = true;
} 

Reference::To<polyco> dsp::Fold::get_folding_polyco(const psrephem* pephemeris,
						    const Observation* observation){
  Reference::To<MatchingPolyco> mpoly = new MatchingPolyco;
  
  MatchingEphemeris* mephem = dynamic_cast<MatchingEphemeris*>(const_cast<psrephem*>(pephemeris));
  if( mephem )
    mpoly->set_matching_sources( mephem->get_matching_sources() );
  
  polycos.push_back( mpoly.get() );
  
  Reference::To<polyco> the_folding_polyco = mpoly.get();
  
  MJD time = observation->get_start_time();

  Tempo::set_polyco ( *the_folding_polyco.get(), *pephemeris, time, time,
		      nspan, ncoef, 8, observation->get_telescope_code() );
  
  return the_folding_polyco;
}

polyco* dsp::Fold::choose_polyco (const MJD& time, const string& pulsar)
{
  if (verbose) cerr << "dsp::Fold::choose_polyco checking "
		    << polycos.size()
		    << " specified polycos" << endl;

  for (unsigned ipoly=0; ipoly<polycos.size(); ipoly++){

    MatchingPolyco* mpoly = dynamic_cast<MatchingPolyco*>(polycos[ipoly].ptr());

    if( mpoly && mpoly->matches(time,pulsar) )
      return polycos[ipoly];
    else if( polycos[ipoly]->i_nearest(time,pulsar) >= 0 ){
      if (verbose)
	cerr << "PSR: " << pulsar << " found in polyco entry\n";
      return polycos[ipoly];
    }

  }

  return 0;
}

psrephem* dsp::Fold::choose_ephemeris (const string& pulsar)
{
  if (verbose) cerr << "dsp::Fold::choose_ephemeris checking "
		    << ephemerides.size()
		    << " specified ephemerides" << endl;

  for (unsigned ieph=0; ieph<ephemerides.size(); ieph++) {

    if (verbose) cerr << "dsp::Fold::prepare compare " 
		      << pulsar << " and "
		      << ephemerides[ieph]->psrname() << endl;

    MatchingEphemeris* mephem = dynamic_cast<MatchingEphemeris*>(ephemerides[ieph].ptr());

    if( mephem && mephem->matches(pulsar) )
      return ephemerides[ieph];
    else if (pulsar.find (ephemerides[ieph]->psrname()) != string::npos) {
      if (verbose)
	cerr << "PSR: " << pulsar << " matches parfile entry\n";
      return ephemerides[ieph];
    }
  }

  return 0;
}


/*! Unless over-ridden by calling the Fold::set_nbin method, this attribute
  determines the maximum number of pulse phase bins into which input data
  will be folded. */
unsigned dsp::Fold::maximum_nbin = 8192;

/*! The minimum width of each pulse phase bin is specified in units of
  the time resolution of the input TimeSeries. */
double dsp::Fold::minimum_bin_width = 1.5;

/*! If true, the number of bins chosen by Fold::choose_nbin will be a
  power of two.  If false, there is no constraint on the value returned. */
bool dsp::Fold::power_of_two = true;

/*! Based on both the period of the signal to be folded and the time
  resolution of the input TimeSeries, this method calculates a
  sensible number of bins into which the input data will be folded.
  The value returned by this method may be over-ridden by calling the
  Fold::set_nbin method.  The maximum number of bins to be used during
  folding may be set through the Fold::maximum_nbin attribute. */
unsigned dsp::Fold::choose_nbin ()
{
  double folding_period = get_folding_period ();

  if (verbose)
    cerr << "dsp::Fold::choose_nbin folding_period=" << folding_period << endl;

  if (folding_period == 0.0)
    throw Error (InvalidState, "dsp::Fold::choose_nbin",
		 "no folding period or polyco set");

  double sampling_period = 1.0 / input->get_rate();
  double binwidth = minimum_bin_width * sampling_period;

  unsigned sensible_nbin = unsigned (folding_period / binwidth);

  if (power_of_two) {
    // make sensible_nbin the largest power of two less than the maximum  
    sensible_nbin = (unsigned)
      pow ( 2.0, floor(log(folding_period/binwidth)/log(2.0)) );
  }

  if (requested_nbin > 1) {

    // the Fold::set_nbin method has been called

    if (verbose) cerr << "dsp::Fold::choose_nbin using requested nbin="
		      << requested_nbin << endl;

    if (requested_nbin > sensible_nbin) {
      cerr << "dsp::Fold::choose_nbin WARNING Requested nbin=" 
	   << requested_nbin << " > sensible nbin=" << sensible_nbin << "."
	"  Where:\n"
	"  sampling period     = " << sampling_period*1e3 << " ms and\n"
	"  requested bin width = " << folding_period/requested_nbin*1e3 << 
	" ms\n" << endl;
    }

    folding_nbin = requested_nbin;

  }
  else {
    // the Fold::set_nbin method has not been called.  choose away ...

    if (maximum_nbin && sensible_nbin > maximum_nbin) {

      if (verbose) cerr << "dsp::Fold::choose_nbin using maximum nbin=" 
			<< maximum_nbin << endl;
      folding_nbin = maximum_nbin;

    }
    else {

      if (verbose) cerr << "dsp::Fold::choose_nbin using sensible nbin=" 
			<< sensible_nbin << endl;
      folding_nbin = sensible_nbin;

    }

  }

  return folding_nbin;
}

//! Set the reference phase (phase of bin zero)
void dsp::Fold::set_reference_phase (double phase)
{
  // ensure that phase runs from 0 to 1
  phase -= floor (phase);
  reference_phase = phase;
}

//! Set the period at which to fold data (in seconds)
void dsp::Fold::set_folding_period (double _folding_period)
{
  folding_period_source = string();
  folding_period = _folding_period;
  folding_polyco = 0;
  built = true;
}

//! Set the period at which to fold data, but only do it for this source (in seconds)
void dsp::Fold::set_folding_period (double folding_period, string _folding_period_source){
  set_folding_period( folding_period );
  folding_period_source = _folding_period_source;
}

//! Get the average folding period
double dsp::Fold::get_folding_period () const
{
  if (folding_polyco)
    return folding_polyco->get_refperiod();
  else
    return folding_period;
}

//! Set the phase polynomial(s) with which to fold data
void dsp::Fold::set_folding_polyco (const polyco* _folding_polyco)
{
  folding_polyco = _folding_polyco;
  folding_period = 0.0;
  built = true;
}

const polyco* dsp::Fold::get_folding_polyco () const
{
  return folding_polyco;
}

void dsp::Fold::set_ncoef (unsigned _ncoef)
{
  if (ncoef == _ncoef)
    return;

  ncoef = _ncoef; 
  built = false;
}

void dsp::Fold::set_nspan (unsigned _nspan)
{
  if (nspan == _nspan)
    return;

  nspan = _nspan;
  built = false;
}

void dsp::Fold::set_pulsar_ephemeris (const psrephem* ephemeris)
{
  if (pulsar_ephemeris == ephemeris)
    return;

  pulsar_ephemeris = ephemeris;
  built = false;
}

const psrephem* dsp::Fold::get_pulsar_ephemeris () const
{
  return pulsar_ephemeris.ptr();
}

void dsp::Fold::set_input (TimeSeries* _input)
{
  if (verbose)
    cerr << "dsp::Fold::set_input (TimeSeries* =" << _input << ")" << endl;

  Transformation<const TimeSeries, PhaseSeries>::set_input (_input);

  weighted_input = dynamic_cast<WeightedTimeSeries*> (_input);

  if (verbose && weighted_input)
    cerr << "dsp::Fold::set_input input is a WeightedTimeSeries" << endl;
}

//! Add a phase model with which to choose to fold the data
void dsp::Fold::add_folding_polyco (polyco* folding_polyco){
  if( folding_polyco )
    polycos.push_back( folding_polyco );
}

//! Add an ephemeris with which to choose to create the phase model
void dsp::Fold::add_pulsar_ephemeris (psrephem* pulsar_ephemeris){
  if( pulsar_ephemeris )
    ephemerides.push_back( pulsar_ephemeris );
}

void dsp::Fold::transformation ()
{
  if (!input->get_detected ())
    throw Error (InvalidParam, "dsp::Fold::transformation",
		 "input is not detected");

  if (!built)
    prepare ();

  if (folding_period == 0 && !folding_polyco)
    throw Error (InvalidState, "dsp::Fold::transformation",
		 "no folding period or polyco set");

  if (folding_nbin == 0)
    choose_nbin ();

  if ( output->integration_length &&
       output->get_reference_phase() != get_reference_phase() )
    throw Error (InvalidState, "dsp::Fold::transformation",
		 "output reference phase=%lf != reference phase=%lf",
		 output->get_reference_phase(), get_reference_phase() );

  // Temporarily make sure the DMs are the same
  if( pulsar_ephemeris || dispersion_measure >= 0 )
    output->set_dispersion_measure( input->get_dispersion_measure() ); 

  if (verbose)
    cerr << "dsp::Fold::transformation call PhaseSeries::mixable" << endl;

  if (!output->mixable (*input, folding_nbin, idat_start, ndat_fold)){
    string input_str;
    string output_str;

    input->obs2string(input_str);
    output->obs2string(output_str);

    fprintf(stderr,"Input:\n\n%s\n\n",input_str.c_str());
    fprintf(stderr,"Ouput:\n\n%s\n\n",output_str.c_str());
	    
    throw Error (InvalidParam, "dsp::Fold::transformation",
		 "input and output are not PhaseSeries::mixable");
  }

  unsigned blocks = input->get_nchan() * input->get_npol();

  uint64 block_size = input->get_datptr (0,1) - input->get_datptr (0,0);
  uint64 block_ndat = block_size / input->get_ndim();

  assert (block_size % input->get_ndim() == 0);

  double integrated = 0.0;

  const unsigned* weights = 0;
  unsigned ndatperweight = 0;

  if (weighted_input) {
    if (verbose)
      cerr << "dsp::Fold::transformation WeightedTimeSeries" << endl;
    weights = weighted_input->get_weights();
    ndatperweight = weighted_input->get_ndat_per_weight();
  }

  fold (integrated, output->get_datptr(), &(output->hits[0]),
	input, blocks, input->get_datptr(), block_ndat, input->get_ndim(),
	weights, ndatperweight, idat_start, ndat_fold);

  output->integration_length += integrated;
  
  if (folding_period)
    output->set_folding_period( folding_period );
  else
    output->set_pulsar_ephemeris( pulsar_ephemeris, folding_polyco );

  output->set_reference_phase( reference_phase );

  // set the sampling rate of the output PhaseSeries
  double sampling_interval = pfold / double(folding_nbin);
  output->set_rate (1.0/sampling_interval);

  // Set things out of the pulsar ephemeris
  if( dispersion_measure > 0.0 ){
    output->set_dispersion_measure( dispersion_measure );
    fprintf(stderr,"folder has set DM to be %f\n",dispersion_measure);
  }
  else if( pulsar_ephemeris )
    output->set_dispersion_measure( pulsar_ephemeris->get_dm() );

  if( archive_filename.size() > 0 ){
    output->set_archive_filename( archive_filename );
    fprintf(stderr,"dsp::Fold Have set archive_filename to '%s'\n",archive_filename.c_str());
  }

  if( archive_filename_extension.size() > 0 ){
    output->set_archive_filename_extension( archive_filename_extension );
    fprintf(stderr,"dsp::Fold yayayaya Have set archive_filename_extension to '%s'\n",archive_filename_extension.c_str());
  }

}

/*!  This method creates a folding plan and then folds nblock arrays.

   \pre the folding_nbin and folding_period or folding_polyco attributes must
   have been set prior to calling this method.

   \param info Observation telling the start_time and sampling_rate of time
   \param nblock the number of blocks of data to be folded
   \param integration returns the time integrated
   \param phase base address of nblock contiguous phase blocks (of nbin*ndim)
   \param hits array of nbin phase bin counts
   \param time base address of nblock contiguous time blocks (of ndat*ndim)
   \param ndat the number of time samples in each time block
   \param ndim the dimension of each time sample
   \param weights corresponding to each block of ndatperweight time samples
   \param ndatperweight number of time samples per weight
   \param idat_start the time sample at which to start folding (optional)
   \param fold_ndat the number of time samples to be folded (optional)
*/
void dsp::Fold::fold (double& integration_length, float* phase, unsigned* hits,
		      const Observation* info, unsigned nblock,
		      const float* time, uint64 ndat, unsigned ndim,
		      const unsigned* weights, unsigned ndatperweight,
		      uint64 idat_start, uint64 ndat_fold)
{
  // /////////////////////////////////////////////////////////////////////////
  //
  // Initialize and check state
  //

  if (!folding_nbin) {

    if (!requested_nbin)
      throw Error (InvalidState, "dsp::Fold::fold", "nbin not set");

    folding_nbin = requested_nbin;
  }

  if (!folding_polyco && !folding_period)
    throw Error (InvalidState, "dsp::Fold::fold",
		 "no polynomial and no period specified");

  if (ndat_fold == 0)
    ndat_fold = ndat;

  uint64 idat_end = idat_start + ndat_fold;

  if (idat_end > ndat)
    throw Error (InvalidParam, "dsp::Fold:fold",
		 "idat_start="UI64" + ndat_fold="UI64" > ndat="UI64,
		 idat_start, ndat_fold, ndat);

  if (verbose)
    cerr << "dsp::Fold::fold ndat_fold=" << ndat_fold << endl;

  // interval of each time sample in seconds
  double sampling_interval = 1.0/info->get_rate();

  // midpoint of the first sample
  double mid_idat_start = double(idat_start) + 0.5;

  // MJD of midpoint of the first sample
  MJD start_time = info->get_start_time() + mid_idat_start * sampling_interval;

  double phi=0;
  pfold = 0;

  // /////////////////////////////////////////////////////////////////////////
  //
  // Calculate phase gradient across this section of data
  //

  if (folding_period != 0) {

    pfold = folding_period;
    phi   = fmod (start_time.in_seconds(), folding_period) / folding_period;

    if (verbose)
      cerr << "dsp::Fold::fold constant folding period=" << pfold << endl;

  }
  else {
    // find the period and phase at the mid time of the first sample
    pfold = folding_polyco->period(start_time);
    phi   = folding_polyco->phase(start_time).fracturns();

    if (verbose)
      cerr << "dsp::Fold::fold polyco.period=" << pfold << endl;

  }

  // adjust to reference phase
  phi -= reference_phase;

  if (verbose)
    cerr << "dsp::Fold::fold phase=" << phi << endl;

  // /////////////////////////////////////////////////////////////////////////
  //
  // Construct a folding plan
  //

  // allocate storage for phase bin plan
  unsigned* binplan = (unsigned*) workingspace (ndat_fold * sizeof(unsigned));

  // index through time dimension
  uint64 idat = idat_start;
  // index through phase dimension
  unsigned ibin = 0;
  // index through space dimension
  unsigned idim = 0;
  // index through weight array
  unsigned iweight = 0;
  // idat of last point in current weight
  unsigned long datendweight = 0;

  // number of time samples actually folded
  uint64 ndat_folded = 0;

  if (ndatperweight) {
    iweight = idat_start / ndatperweight;
    datendweight = (iweight + 1) * ndatperweight;

    if (weights[iweight] == 0)
      discarded_weights ++;

  }

  double double_nbin = double (folding_nbin);
  double phase_per_sample = sampling_interval / pfold;

  for (idat=idat_start; idat < idat_end; idat++) {

    if (ndatperweight && idat >= datendweight) {
      iweight ++;
      datendweight += ndatperweight;

      if (weights[iweight] == 0)
        discarded_weights ++;

    }

    // ensure that phi runs from 0 to 1
    phi -= floor (phi);

    ibin = unsigned (phi * double_nbin);

    phi += phase_per_sample;

    assert (ibin < folding_nbin);
    binplan[idat-idat_start] = ibin;

    if (!ndatperweight || weights[iweight] != 0) {
      hits[ibin]++;
      ndat_folded ++;
    }

  }

  // /////////////////////////////////////////////////////////////////////////
  //
  // Calculate the integrated total
  //
  integration_length = double(ndat_folded) * sampling_interval;
  if (verbose)
    cerr << "dsp::Folding::fold " << integration_length << " seconds" << endl;

  
  // /////////////////////////////////////////////////////////////////////////
  //
  // Fold arrays
  //

  // pointer through phase dimension
  float *phasep;
  // pointer through dimensions after phase
  float *phdimp;

  // pointer through time dimension
  const float *timep;
  
  for (unsigned iblock=0; iblock<nblock; iblock++) {

    timep = time + (ndat * iblock + idat_start) * ndim;
    phasep = phase + folding_nbin * iblock * ndim;

    for (idat=0; idat < ndat_fold; idat++) {

      // point to the right phase
      phdimp = phasep + binplan[idat] * ndim;

      // integrate the ndim dimensions
      for (idim=0; idim<ndim; idim++) {
	phdimp[idim] += *timep;
	timep ++;
      }
      
    }
  }

}
