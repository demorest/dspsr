#include <assert.h>

#include "dsp/Fold.h"
#include "dsp/WeightedTimeSeries.h"

#include "Error.h"

#include "psrephem.h"
#include "tempo++.h"
#include "genutil.h"

dsp::Fold::Fold () 
  : Transformation <const TimeSeries, PhaseSeries> ("Fold", outofplace) 
{
  folding_period = 0;

  nbin = 0;
  ncoef = 15;
  nspan = 120;

  built = false;
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
  string jpulsar = observation->get_source();

  if (jpulsar.length() == 0)
    throw Error (InvalidParam, "dsp::Fold::prepare",
		 "empty Observation::source");

  if (jpulsar[0] != 'J')
    jpulsar = "J" + jpulsar;

  if (verbose)
    cerr << "dsp::Fold::prepare source=" << jpulsar << endl;

  MJD time = observation->get_start_time();

  folding_polyco = choose_polyco (time, jpulsar);

  if (folding_polyco)
    return;

  // no polyco found in list of supplied polycos
  pulsar_ephemeris = choose_ephemeris (jpulsar);

  if (!pulsar_ephemeris) {
    Reference::To<psrephem> ephemeris = new psrephem;

    if (ephemeris->create (jpulsar, 0) < 0)
      throw Error (FailedCall, "dsp::Fold::prepare",
		   "error psrephem::create ("+jpulsar+")");

    pulsar_ephemeris = ephemeris;
  }

#if 0

  TODO - ephemeris and coordinates in archive

  dm = ephemeris.get_dm();
  if (verbose)
    cerr << "dsp::Fold::prepare psrephem dm = " << dm << endl;

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

  Reference::To<polyco> polly = new polyco;

  Tempo::set_polyco ( *polly, *pulsar_ephemeris, time, time,
		      nspan, ncoef, 8, observation->get_telescope() );

  folding_polyco = polly;

#if 0
  doppler = 1.0 + psr_poly->doppler_shift(raw.start_time);
  
  if (verbose)
    cerr << "dsp::Fold::prepare Doppler shift from polyco:" << doppler << endl;

#endif

  built = true;
} 

polyco* dsp::Fold::choose_polyco (const MJD& time, const string& pulsar)
{
  if (verbose) cerr << "dsp::Fold::choose_polyco checking "
		    << polycos.size()
		    << " specified polycos" << endl;

  for (unsigned ipoly=0; ipoly<polycos.size(); ipoly++)

    if (polycos[ipoly]->nearest (time, pulsar)
	&& polycos[ipoly]->nearest (time, pulsar)) {
      if (verbose)
	cerr << "PSR: " << pulsar << " found in polyco entry\n";
      return polycos[ipoly];
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

    if (pulsar.find (ephemerides[ieph]->psrname()) != string::npos) {
      if (verbose)
	cerr << "PSR: " << pulsar << " matches parfile entry\n";
      return ephemerides[ieph];
    }
  }

  return 0;
}

//! Set the period at which to fold data (in seconds)
void dsp::Fold::set_folding_period (double _folding_period)
{
  folding_period = _folding_period;
  folding_polyco = 0;
  built = true;
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
  if( !pulsar_ephemeris )
    return NULL;

  return pulsar_ephemeris;
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

void dsp::Fold::transformation ()
{
  if (nbin == 0)
    throw Error (InvalidState, "dsp::Fold::transformation", "nbin not set");

  if (!input->get_detected ())
    throw Error (InvalidParam, "dsp::Fold::transformation",
		 "input is not detected");

  if (!built)
    prepare ();

  if (folding_period == 0 && !folding_polyco)
    throw Error (InvalidState, "dsp::Fold::transformation",
		 "no folding period or polyco set");

  if (verbose)
    cerr << "dsp::Fold::transformation call PhaseSeries::mixable" << endl;

  if (!output->mixable (*input, nbin))
    throw Error (InvalidParam, "dsp::Fold::transformation",
		 "input and output are not PhaseSeries::mixable");

  unsigned blocks = input->get_nchan() * input->get_npol();

  uint64 block_size = input->get_datptr (0,1) - input->get_datptr (0,0);
  uint64 block_ndat = block_size / input->get_ndim();

  assert (block_size % input->get_ndim() == 0);

  double integrated = 0.0;

  const unsigned* weights = 0;
  unsigned ndatperweight = 0;

  if (weighted_input) {
    weights = weighted_input->get_weights();
    ndatperweight = weighted_input->get_ndat_per_weight();
  }

  fold (integrated, output->get_datptr(), output->hits.begin(),
	input, blocks, input->get_datptr(), block_ndat, input->get_ndim(),
	weights, ndatperweight);

  output->integration_length += integrated;
  
  if (folding_period)
    output->set_folding_period( folding_period );
  else
    output->set_folding_polyco( folding_polyco );

}

/*!  This method creates a folding plan and then folds nblock arrays.

   \pre the nbin and folding_period or folding_polyco attributes must
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

  if (!nbin)
    throw Error (InvalidState, "dsp::Fold::fold", "nbin not set");

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

  double phi=0, pfold=0;

  // /////////////////////////////////////////////////////////////////////////
  //
  // Calculate phase gradient across this section of data
  //

  if (folding_period != 0) {

    pfold = folding_period;
    phi   = fmod (start_time.in_seconds(), folding_period) / folding_period;
    while (phi<0.0) phi += 1.0;

    if (verbose)
      cerr << "dsp::Fold::fold CAL period=" << pfold << endl;

  }
  else {

    // find the period and phase at the mid time of the first sample
    pfold = folding_polyco->period(start_time);
    phi = folding_polyco->phase(start_time).fracturns();
    if (phi<0.0) phi += 1.0;
    
    if (verbose)
      cerr << "dsp::Fold::fold polyco.period=" << pfold << endl;

  }

  double nphi = double (nbin);
  double binspersample = nphi * sampling_interval / pfold;
  phi *= nphi;

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
  }

  for (idat=idat_start; idat < idat_end; idat++) {

    if (idat >= datendweight) {
      iweight ++;
      datendweight += ndatperweight;
    }

    ibin = unsigned(phi);
    phi += binspersample;
    if (phi >= nphi) phi -= nphi;

    assert (ibin < nbin);
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
    phasep = phase + nbin * iblock * ndim;

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
