/***************************************************************************
 *
 *   Copyright (C) 2002-2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Fold.h"
#include "dsp/ObservationChange.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/Scratch.h"

#include "Pulsar/ParametersLookup.h"
#include "Predict.h"
#include "Error.h"

#include <assert.h>

using namespace std;

dsp::Fold::Fold () :
  Transformation <TimeSeries, PhaseSeries> ("Fold", outofplace) 
{
  folding_period = 0;

  requested_nbin = 0;
  folding_nbin = 0;
  force_sensible_nbin = false;

  reference_phase = -1.0; // Defaults to 0.0

  ncoef = 0; // Defaults to 15 during initialise
  nspan = 0; // Defaults to 120 during initialise

  built = false;

  idat_start = ndat_fold = 0;
}

void dsp::Fold::set_engine (Engine* _engine)
{
  engine = _engine;

  if (engine)
  {
    engine->set_parent (this);
    engine->set_cerr (this->cerr);
  }
}

void dsp::Fold::set_cerr (ostream& os) const
{
  if (engine)
    engine->set_cerr(os);
  Transformation<TimeSeries,PhaseSeries>::set_cerr(os);
}

//! Set any unititialized parameters
void dsp::Fold::initialise()
{
  if( ncoef==0 )
    ncoef = 15;

  if( nspan==0 ){
    nspan = 120;   
    if (psrdisp_compatible) {
      cerr << "dsp::Fold psrdisp compatibility\n"
        "   using nspan of 960 instead of 120" << endl;
      nspan = 960;
    }
  }

  if( reference_phase < 0.0 )
    reference_phase = 0.0;
}

dsp::Fold::~Fold ()
{
}

dsp::Fold* dsp::Fold::clone () const
{
  return new Fold(*this);
}

dsp::PhaseSeries* dsp::Fold::get_output () const
{
  if (engine)
    return engine->get_profiles ();

  return output;
}

//! Prepare for folding the input TimeSeries
void dsp::Fold::prepare ()
{
  if (!input)
    throw Error (InvalidState, "dsp::Fold::prepare", "no input");

  prepare (input);
}

//! Combine results with another operation
void dsp::Fold::combine (const Operation* other)
{
  Operation::combine (other);

  const Fold* fold = dynamic_cast<const Fold*>( other );
  if (!fold)
    return;

  if (verbose)
    cerr << "dsp::Fold::combine another Fold" << endl;

  get_result()->combine( fold->get_result() );

  if (verbose)
    cerr << "dsp::Fold::combine another Fold exit" << endl;
}

dsp::PhaseSeries* dsp::Fold::get_result () const
{
  if (engine)
    engine->synch (output);

  return output;
}

void dsp::Fold::reset ()
{
  if (verbose)
    cerr << "dsp::Fold::reset" << endl;

  Operation::reset ();

  if (engine)
    engine->zero();
  if (output)
    output->zero();
}

void dsp::Fold::finish ()
{
  get_result();
}

//! Prepare for folding the given Observation
void dsp::Fold::prepare (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::Fold::prepare" << endl;

  initialise();

  // The Fold::prepare method may be called with an Observation state
  // not equal to the state of the input to be folded.  Therefore, the
  // choice of folding_nbin is postponed until the first call to
  // Fold::transformation.

  folding_nbin = 0;

  Reference::To<Observation> copy;
  if (change)
  {
    copy = new Observation (*observation);
    change->change(copy);
    observation = copy;
  }

  string pulsar = observation->get_source();

  if (pulsar.length() == 0)
    throw Error (InvalidParam, "dsp::Fold::prepare", "empty source name");

  // Is this a CAL?
  if (observation->get_type() == Signal::PolnCal)
  {
    double calperiod = 1.0/observation->get_calfreq();
    set_folding_period (calperiod);
  }

  if (folding_period > 0)
  {
    if (verbose)
      cerr << "dsp::Fold::prepare using folding_period="
           << folding_period << endl;
    pulsar_ephemeris = 0;
    folding_predictor = 0;
    built = true;
    return;
  }

  if (folding_predictor)
  {
    if (verbose)
      cerr << "dsp::Fold::prepare using given predictor" << endl;
    built = true;
    return;
  }

  if (!pulsar_ephemeris)
  {
    if (verbose)
      cerr << "dsp::Fold::prepare generating ephemeris" << endl;

    Pulsar::Parameters::Lookup lookup;
    pulsar_ephemeris = lookup (pulsar);
  }

  if (verbose)
    cerr << "dsp::Fold::prepare creating predictor" << endl;

  folding_predictor = get_folding_predictor (pulsar_ephemeris, observation);

  built = true;
} 

Pulsar::Predictor*
dsp::Fold::get_folding_predictor (const Pulsar::Parameters* params,
                                  const Observation* observation)
{
  MJD time = observation->get_start_time()-0.01;
  Pulsar::Generator* generator = Pulsar::Generator::factory (params);

  Tempo::Predict* predict = dynamic_cast<Tempo::Predict*>( generator );
  if (predict)
  {
    predict->set_nspan ( nspan );
    predict->set_ncoef ( ncoef );    
  }

  /*
   * Tempo2 predictor code:
   *
   * Here we make a predictor valid for the next 24 hours
   * I consider this to be a bit of a hack, since theoreticaly
   * observations could be longer, and it's a bit silly to make
   * such a polyco for a 10 min obs.
   *
   */
  MJD endtime = time + 86400;

  generator->set_site( observation->get_telescope() );
  generator->set_parameters( params );
  generator->set_time_span( time, endtime );

  double freq = observation->get_centre_frequency();
  double bw = fabs( observation->get_bandwidth() );
  generator->set_frequency_span( freq-bw/2, freq+bw/2 );

  if (verbose)
    cerr << "dsp::Fold::get_folding_predictor"
            " calling Pulsar::Generator::generate" << endl;

  return generator->generate ();
}




/*! Unless over-ridden by calling the Fold::set_nbin method, this attribute
  determines the maximum number of pulse phase bins into which input data
  will be folded. */
unsigned dsp::Fold::maximum_nbin = 1024;

/*! The minimum width of each pulse phase bin is specified in units of
  the time resolution of the input TimeSeries. */
double dsp::Fold::minimum_bin_width = 1.2;

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
  if (verbose)
    cerr << "dsp::Fold::choose_nbin" << endl;

  double the_folding_period = get_folding_period ();

  if (verbose)
    cerr << "dsp::Fold::choose_nbin folding_period=" 
         << the_folding_period << endl;

  if (the_folding_period <= 0.0)
  {
    throw Error (InvalidState, "dsp::Fold::choose_nbin",
                 "no folding period or Pulsar::Predictor set. eph=%p",
                 pulsar_ephemeris.ptr());
  }
  double sampling_period = 1.0 / input->get_rate();

  if (verbose)
    cerr << "dsp::Fold::choose_nbin sampling_period=" 
         << sampling_period << endl;

  if (sampling_period < 0)
    throw Error (InvalidState, "dsp::Fold::choose_nbin",
                 "input has a negative sampling rate");

  double binwidth = minimum_bin_width * sampling_period;

  if (verbose)
    cerr << "dsp::Fold::choose_nbin minimum_bin_width=" 
         << minimum_bin_width << " bins or " << binwidth << " seconds"
         << endl;

  unsigned sensible_nbin = unsigned (the_folding_period / binwidth);

  if (verbose)
    cerr << "dsp::Fold::choose_nbin sensible nbin=" 
         << sensible_nbin << endl;

  if (power_of_two) {
    double log2bin = log(the_folding_period/binwidth) / log(2.0);
    // make sensible_nbin the largest power of two less than the maximum  
    sensible_nbin = (unsigned) pow (2.0, floor(log2bin));

    if (verbose)
      cerr << "dsp::Fold::choose_nbin largest power of 2 < nbin=" 
           << sensible_nbin << endl;
  }

  if (sensible_nbin == 0) {
    cerr << "dsp::Fold::choose_nbin WARNING no phase resolution\n"
        "  sampling period = " << sampling_period*1e3 << " ms and\n"
        "  folding period  = " << the_folding_period*1e3 << " ms" << endl;
    sensible_nbin = 1;
  }

  if (requested_nbin > 1) {
    // the Fold::set_nbin method has been called
    if (verbose) cerr << "dsp::Fold::choose_nbin using requested nbin="
                      << requested_nbin << endl;
    folding_nbin = requested_nbin;

    if (requested_nbin > sensible_nbin) {
      if (force_sensible_nbin){
         // if we are forcing sensible bins, change nbin.
         folding_nbin=sensible_nbin;
      } else {
         // otherwise tell the user they are being rather foolish/optimistic!
        cerr << "dsp::Fold::choose_nbin WARNING Requested nbin=" 
           << requested_nbin << " > sensible nbin=" << sensible_nbin << "."
        "  Where:\n"
        "  sampling period     = " << sampling_period*1e3 << " ms and\n"
        "  requested bin width = " << the_folding_period/requested_nbin*1e3 << 
        " ms\n" << endl;
      }
    }
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
  folding_period = _folding_period;
  folding_predictor = 0;
  built = true;
}

//! Set the name of the source
void dsp::Fold::set_change (const ObservationChange* c)
{
  change = c;
  built = false;
}

#if 0
//! Get the name of the source
std::string dsp::Fold::get_source_name () const
{
  if (input)
  {
    const Observation* observation = input;

    Reference::To<Observation> copy;
    if (change)
    {
      copy = new Observation (&observation);
      change->change(copy);
      observation = copy;
    }

    return observation->get_source();
  }

  if (pulsar_ephemeris)
    return pulsar_ephemeris->get_name();

  return "";
}
#endif

//! Get the average folding period
double dsp::Fold::get_folding_period () const
{
  if (folding_predictor)
    return 1.0/folding_predictor->frequency(input->get_start_time());
  else
    return folding_period;
}

//! Set the phase polynomial(s) with which to fold data
void dsp::Fold::set_folding_predictor (const Pulsar::Predictor* _folding_predictor)
{
  folding_predictor = _folding_predictor;
  folding_period = 0.0;
  built = true;
}

const Pulsar::Predictor* dsp::Fold::get_folding_predictor () const
{
  return folding_predictor;
}

bool dsp::Fold::has_folding_predictor () const
{
  return folding_predictor;
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

void dsp::Fold::set_pulsar_ephemeris (const Pulsar::Parameters* ephemeris)
{
  if (pulsar_ephemeris == ephemeris)
    return;

  pulsar_ephemeris = ephemeris;
  built = false;
}

const Pulsar::Parameters* dsp::Fold::get_pulsar_ephemeris () const
{
  return pulsar_ephemeris;
}

void dsp::Fold::set_input (const TimeSeries* _input)
{
  if (verbose)
    cerr << "dsp::Fold::set_input (TimeSeries* =" << _input << ")" << endl;

  Transformation<TimeSeries, PhaseSeries>::set_input (_input);

  weighted_input = dynamic_cast<const WeightedTimeSeries*> (_input);

  if (verbose && weighted_input)
    cerr << "dsp::Fold::set_input input is a WeightedTimeSeries" << endl;
}

void dsp::Fold::check_input() try
{
  if (!input->get_detected ())
    throw Error (InvalidParam, "dsp::Fold::check_input",
		 "input is not detected");
}
catch (Error &error) 
{
  throw error += "dsp::Fold::check_input";
}

void dsp::Fold::prepare_output() try
{
  if (verbose)
    cerr << "dsp::Fold::prepare_output call PhaseSeries::mixable" << endl;

  if (!get_output()->mixable (*input, folding_nbin, idat_start, ndat_fold))
    throw Error (InvalidParam, "dsp::Fold::prepare_output",
		 "input and output are not mixable " 
                 + get_output()->get_reason());
}
catch (Error &error)
{
  throw error += "dsp::Fold::prepare_output()";
}

void dsp::Fold::transformation () try
{
  if (verbose)
    cerr << "dsp::Fold::transformation" << endl;

  if (input->get_ndat() == 0)
    return;

  if (!built)
  {
    if (verbose)
      cerr << "dsp::Fold::transformation prepare" << endl;
    prepare ();
  }

  if (folding_period <= 0 && !folding_predictor)
    throw Error (InvalidState, "dsp::Fold::transformation",
                 "no folding period or Pulsar::Predictor set");

  if (folding_nbin == 0)
  {
    if (verbose)
      cerr << "dsp::Fold::transformation choose_nbin" << endl;
    choose_nbin ();
  }

  PhaseSeries* use = get_output();

  if ( use->integration_length &&
       use->get_reference_phase() != get_reference_phase() )
    throw Error (InvalidState, "dsp::Fold::transformation",
                 "output reference phase=%lf != reference phase=%lf",
                 use->get_reference_phase(), get_reference_phase() );

  // Temporarily make sure the DMs are the same
  use->set_dispersion_measure( input->get_dispersion_measure() ); 

  if (verbose)
    cerr << "dsp::Fold::transformation call Fold::prepare_output" << endl;

  set_limits (input);

  prepare_output();

  uint64_t nweights = 0;
  const unsigned* weights = 0;
  unsigned ndatperweight = 0;
  unsigned weight_idat = 0;

  if (weighted_input)
  {
    nweights = weighted_input->get_nweights();
    weights = weighted_input->get_weights();
    ndatperweight = weighted_input->get_ndat_per_weight();
    weight_idat = unsigned(weighted_input->get_weight_idat());

    if (verbose)
      cerr << "dsp::Fold::transformation WeightedTimeSeries weights="
           << weights << " ndatperweight=" << ndatperweight << endl;
  }

  fold (nweights, weights, ndatperweight, weight_idat);
  
  if (folding_period > 0.0)
    use->set_folding_period( folding_period );
  else
  {
    if (pulsar_ephemeris)
    {
      if (verbose)
        cerr << "dsp::Fold::transformation set output ephemeris" << endl;
      use->set_pulsar_ephemeris( pulsar_ephemeris );
    }

    if (folding_predictor)
    {
      if (verbose)
        cerr << "dsp::Fold::transformation set output predictor" << endl;
      use->set_folding_predictor( folding_predictor );
    }
  }

  use->set_reference_phase( reference_phase );

  // set the sampling rate of the output PhaseSeries
  double sampling_interval = pfold / double(folding_nbin);
  use->set_rate (1.0/sampling_interval);

  if (change)
    change->change (use);
}
catch (Error& error)
{
  throw error += "dsp::Fold::transformation";
}

/*!  This method creates a folding plan and then folds nblock arrays.

   \pre the folding_nbin and folding_period or folding_predictor attributes must
   have been set prior to calling this method.

   \param integration_length returns the time integrated 
   \param phase base address of nblock contiguous phase blocks (of nbin*ndim)
   \param hits array of nbin phase bin counts
   \param info Observation telling the start_time and sampling_rate of time
   \param nblock the number of blocks of data to be folded
   \param time base address of nblock contiguous time blocks (of ndat*ndim)
   \param ndat the number of time samples in each time block
   \param ndim the dimension of each time sample
   \param weights corresponding to each block of ndatperweight time samples
   \param ndatperweight number of time samples per weight
   \param idat_start the time sample at which to start folding (optional)
   \param ndat_fold the number of time samples to be folded (optional)

*/

void dsp::Fold::fold (uint64_t nweights,
                      const unsigned* weights, 
                      unsigned ndatperweight,
                      unsigned weight_idat)
{
  if (!folding_nbin)
  {
    if (!requested_nbin)
      throw Error (InvalidState, "dsp::Fold::fold", "nbin not set");
    folding_nbin = requested_nbin;
  }

  if (!folding_predictor && !folding_period)
    throw Error (InvalidState, "dsp::Fold::fold",
                 "no polynomial and no period specified");

  uint64_t ndat = get_input()->get_ndat();
  uint64_t idat_end = idat_start + ndat_fold;

  if (idat_end > ndat)
    throw Error (InvalidParam, "dsp::Fold:fold",
                 "idat_start="UI64" + ndat_fold="UI64" > ndat="UI64,
                 idat_start, ndat_fold, ndat);

  // midpoint of the first sample
  double mid_idat_start = double(idat_start) + 0.5;
  // MJD of midpoint of the first sample
  MJD start_time = get_input()->get_start_time()
    + mid_idat_start / get_input()->get_rate();

  double phi = get_phi(start_time);
  pfold = get_pfold(start_time);

  // allocate storage for phase bin plan
  unsigned* binplan = scratch->space<unsigned> (ndat_fold);

  // index through weight array
  uint64_t iweight = 0;
  // idat of last point in current weight
  uint64_t idat_nextweight = 0;
  // number of bad weights encountered this run
  unsigned bad_weights = 0;
  // number of weights encountered this run
  unsigned tot_weights = 0;

  // number of time samples folded
  uint64_t ndat_folded = 0;
  uint64_t ndat_not_folded = 0;

  // if the input contains zeroed samples that have been zapped by RFI mitigation
  const bool zeroed_samples = input->get_zeroed_data();

  // unique identifier for this fold (helps with multi-threaded debugging)
  uint64_t id = input->get_input_sample() + idat_start;

  if (verbose)
    cerr << "dsp::Fold::fold " << id << " idat_start=" << idat_start 
         << " ndat_fold=" << ndat_fold << endl;

  bool bad_data = false;

  if (ndatperweight)
  {
    iweight = (idat_start + weight_idat) / ndatperweight;

    idat_nextweight = (iweight + 1) * ndatperweight - weight_idat;

    if (verbose)
      cerr << "dsp::Fold::fold " << id << " ndatperweight=" << ndatperweight 
           << " weight_idat=" << weight_idat << " iweight=" << iweight 
           << " nweights=" << nweights << endl;

    if (iweight >= nweights)
    {
      Error error (InvalidState, "dsp::Fold::fold");
      error << "iweight=" << iweight << " >= nweight=" << nweights << "\n\t"
            << "idat_start=" << idat_start 
            << " weight_idat=" << weight_idat 
            << " ndatperweight=" << ndatperweight;
      throw error;
    }

    tot_weights ++;

    if (!zeroed_samples && (weights[iweight] == 0))
    {
      discarded_weights ++;
      bad_weights ++;
      bad_data = true;
    }
  }

  double sampling_interval = 1.0/get_input()->get_rate();
  double double_nbin = double (folding_nbin);
  double phase_per_sample = sampling_interval / pfold;
  unsigned* hits = get_output()->get_hits();
   
  if (engine)
  {
    if (verbose)
      cerr << "dsp::Fold::fold using engine ptr=" << engine.ptr() << endl;
    engine->set_nbin (folding_nbin);
    engine->set_ndat (idat_end - idat_start, idat_start);
  }

  for (uint64_t idat=idat_start; idat < idat_end; idat++)
  {
    if (ndatperweight && idat >= idat_nextweight)
    {
      iweight ++;
      tot_weights ++;

      assert (iweight < nweights);

      if (!zeroed_samples && (weights[iweight] == 0))
      {
        bad_data = true;
        discarded_weights ++;
        bad_weights ++;
      }
      else
        bad_data = false;

      idat_nextweight += ndatperweight;
    }

    phi -= floor(phi);
    double double_ibin = phi * double_nbin;
    unsigned ibin = unsigned (double_ibin);
    phi += phase_per_sample;

    assert (ibin < folding_nbin);

    if (engine)
      engine->set_bin( idat, double_ibin, phase_per_sample*double_nbin );
    else
      binplan[idat-idat_start] = ibin;

    if (bad_data)
      binplan[idat-idat_start] = folding_nbin;
    else
    {
      if (!zeroed_samples)
      {
        hits[ibin]++;
        ndat_folded ++;
      }
    }
  }

  double time_folded = double(ndat_folded) / get_input()->get_rate();

  if (verbose)
    cerr << "dsp::Fold::fold " << id << " ndat_folded=" << ndat_folded 
         << " time=" << time_folded*1e3 << " ms"
         << " (bad=" << bad_weights << "/" << tot_weights << ")" << endl;

  PhaseSeries* result = get_output();

  result->integration_length += time_folded;
  result->ndat_total += ndat_fold;
  total_weights += tot_weights;

  if ( result->get_nbin() != folding_nbin )
    throw Error (InvalidParam,"dsp::Fold::fold",
                 "folding_nbin != output->nbin (%d != %d)",
                 folding_nbin, result->get_nbin());

  const TimeSeries* in = get_input();

  const unsigned ndim = in->get_ndim();
  const unsigned npol = in->get_npol();
  const unsigned nchan = in->get_nchan();

  if (engine)
  {
    engine->fold ();
    if (zeroed_samples)
      result->integration_length += engine->get_ndat_folded() / get_input()->get_rate();
    return;
  }

  if (verbose)
    cerr << "dsp::Fold::fold ndim=" << ndim << " folding_nbin=" << folding_nbin 
         << " nbin=" << result->get_nbin() << endl;

  if (in->get_order() == TimeSeries::OrderFPT)
  {
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        const float* timep = in->get_datptr(ichan,ipol) + idat_start * ndim;
        float* phasep = result->get_datptr(ichan,ipol);

        for (uint64_t idat=0; idat < ndat_fold; idat++)
        {
          if (binplan[idat] != folding_nbin)
          {
            float* phdimp = phasep + binplan[idat] * ndim;
            for (unsigned idim=0; idim<ndim; idim++)
            {
              phdimp[idim] += timep[idim];
            }
            if (zeroed_samples && ipol==0)
            {
              if (timep[0] != 0)
              {
                hits[binplan[idat]]++;
                ndat_folded++;
              }
              else
              {
                ndat_not_folded++;
              }
            } 
          }
          timep += ndim;
        } // for each idat
      } // for each pol

      if (zeroed_samples && ichan < nchan-1)
        hits += folding_nbin;
    } // for each chan 
  }
  else
  {
    uint64_t nfloat = nchan * npol * ndim;

    const float* timep = in->get_dattfp() + idat_start * nfloat;
    float* phasep = result->get_dattfp();

    for (uint64_t idat=0; idat < ndat_fold; idat++)
    {
      if (binplan[idat] != folding_nbin)
      {
        float* php = phasep + binplan[idat] * nfloat;
        for (unsigned ifloat=0; ifloat<nfloat; ifloat++)
          php[ifloat] += timep[ifloat];
      }
      timep += nfloat;
    }
  }

  if (zeroed_samples)
  {
    time_folded = double (ndat_folded) / (get_input()->get_rate() * nchan);
    result->integration_length += time_folded;
    if (verbose)
    {
      double percent_folded = (double) ndat_folded / (double) (ndat_folded + ndat_not_folded);
      percent_folded *= 100;
      cerr << "dsp::Fold::fold folded " << ndat_folded << " of " 
           << ndat_not_folded + ndat_folded << " " << percent_folded
           << "\% time_folded=" << time_folded << endl;
    }
  }
}

/* changes for omp
  const float* timep;
  float* phasep;
  uint64_t idat;
  float* phdimp;
  unsigned ipol;
  unsigned ichan;

#pragma omp parallel for private(ichan,ipol,timep,phasep,idat,phdimp)   
  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
 for (ipol=0; ipol<npol; ipol++)
    {
      timep = in->get_datptr(ichan,ipol) + idat_start * ndim;

      phasep = result->get_datptr(ichan,ipol);

      for (idat=0; idat < ndat_fold; idat++)
      {
        if (binplan[idat] != folding_nbin)
        {
          phdimp = phasep + binplan[idat] * ndim;
          //for (unsigned idim=0; idim<ndim; idim++)
          //  phdimp[idim] += timep[idim];

          phdimp[0] += timep[0];
          phdimp[1] += timep[1];        
        }
        timep += ndim;
      } // for each idat
    } // for each chan
  } // for each pol
}
*/

double dsp::Fold::get_phi (const MJD& start_time)
{
  if ( folding_period > 0.0 )
    return fmod (start_time.in_seconds(), folding_period) / folding_period 
      - reference_phase;

  return folding_predictor->phase(start_time).fracturns()  - reference_phase;
}

double dsp::Fold::get_pfold (const MJD& start_time)
{
  if ( folding_period > 0.0 )
    return folding_period;
  
  return 1.0/folding_predictor->frequency(start_time);
}

/*! sets idat_start to zero and ndat_fold to input->get_ndat() */
void dsp::Fold::set_limits (const Observation* input)
{
  idat_start = 0;
  ndat_fold = input->get_ndat();
}


void dsp::Fold::Engine::set_parent (Fold* fold)
{
  parent = fold;
}

void dsp::Fold::Engine::setup () try
{
  if (!parent)
    throw Error (InvalidState, "dsp::Fold::Engine::setup",
                 "no parent");

  if (verbose)
    parent->cerr << "dsp::Fold::Engine::setup"
      " parent=" << parent << endl;

  const TimeSeries* in = parent->get_input();

  nchan = in->get_nchan();
  npol = in->get_npol();
  ndim = in->get_ndim();

  input = in->get_datptr(0,0);
  input_span = in->get_nfloat_span();

  PhaseSeries* out = get_profiles();

  output = out->get_datptr(0,0);
  output_span = out->get_nfloat_span();

  hits = out->get_hits();
  hits_nchan = out->get_hits_nchan();
  zeroed_samples = in->get_zeroed_data(); 

  if (verbose)
    parent->cerr << "dsp::Fold::Engine::setup"
    " input=" << input << " span=" << input_span << 
    " output=" << output << " span=" << output_span <<
    " hits=" << hits << " hits_nchan=" << hits_nchan <<
    " zeroed_samples=" << zeroed_samples << endl;
}
 catch (Error& error)
   {
     throw error += "dsp::Fold::Engine::setup";
   }
