#include "Fold.h"
#include "polyco.h"

void dsp::Fold::init ()
{
  integration_length = 0;
  folding_period = 0;
}

//! Set the number of phase bins into which data will be folded
void dsp::Fold::set_nbin (int nbin)
{
  hits.resize(nbin);
}

//! Set the number of phase bins into which data will be folded
int dsp::Fold::get_nbin () const
{
  return (int) hits.size();
}

//! Set the period at which to fold data (in seconds)
void dsp::Fold::set_folding_period (double _folding_period)
{
  folding_period = _folding_period;
  folding_polyco = 0;
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
void dsp::Fold::set_folding_polyco (polyco* _folding_polyco)
{
  folding_polyco = _folding_polyco;
  folding_period = 0.0;
}

//! Get the mid-time of the integration
MJD dsp::Fold::get_midtime () const
{
  MJD midtime = 0.5 * (start_time + end_time);

  if (folding_polyco) {
    // truncate midtime to the nearest pulse phase zero
    Phase phase = folding_polyco->phase(midtime).Floor();
    midtime = folding_polyco->iphase(phase);
  }

  if (folding_period) {
    double phase = fmod (midtime.in_seconds(), folding_period)/folding_period;
    midtime -= phase * folding_period;
  }

  return midtime;
}

//! Get the number of seconds integrated into the profile(s)
double dsp::Fold::get_integration_length () const
{
  return integration_length;
}

//! Reset all phase bin totals to zero
void dsp::Fold::zero ()
{
  integration_length = 0.0;

  unsigned ipt=0; 
  for (ipt=0; ipt<hits.size(); ipt++)
    hits[ipt]=0;

  for (ipt=0; ipt<amps.size(); ipt++)
    amps[ipt]=0.0;
}



// //////////////////////////////////////////////////////////////////////////
//
// how to distinguish between profile count > boundary -> return false;
// and  

bool rawprofile::bound (const observation& obs,
			unsigned long& istart, unsigned long& fndat,
			unsigned& subint_count,
			const vector<MJD>& pulse_boundaries,
			bool& obs_extends_next,
			bool& profile_complete)
{
  obs_extends_next = false;
  profile_complete = false;

  if (subint_count >= pulse_boundaries.size()-1) {
    if (verbose)
      cerr << "rawprofile::bound start subint past boundary" << endl;
    return false;
  }

  MJD fold_start;

  // some comparisons need not be so rigorous for the purposes of this
  // routine.  half the time resolution will do
  double oldMJDprecision = MJD::precision;
  MJD::precision = 0.5/obs.rate;

  if (int_time != 0.0) {
    // check that 'obs' has new data to contribute to this profile

    // first, find the boundaries of the current subint
    for (; subint_count < pulse_boundaries.size()-1; subint_count++)
      if (pulse_boundaries[subint_count+1] > start_time)
	break;

    if ( subint_count >= pulse_boundaries.size()-1 ||
	 !(start_time >= pulse_boundaries[subint_count]) ||
	 !(end_time <= pulse_boundaries[subint_count+1]) ) {
      if (verbose) {
	cerr << "rawprofile::bound subint with start time (" 
	     << start_time << ") outside of boundary!"
	     << endl;
	cerr << "rawprofile::bound boundary:" << pulse_boundaries.front()
	     << " -> " << pulse_boundaries.back()
	     << endl;
      }
      MJD::precision = oldMJDprecision;
      return false;
    }

    if (obs.end_time < pulse_boundaries[subint_count] ||
	obs.start_time > pulse_boundaries[subint_count+1]) {
      if (verbose)
	cerr << "rawprofile::bound obs from different subint" << endl;
      MJD::precision = oldMJDprecision;
      obs_extends_next = true;
      return false;
    }
    
    fold_start = std::max (end_time, obs.start_time);

    if (verbose)
      cerr << "rawprofile::bound start folding at" 
	   << "\n max start=" << fold_start
	   << "\n cur   end=" << end_time
	   << "\n obs start=" << obs.start_time
	   << endl;

  }
  
  else {

    // this subint has no data, start at the first sample of obs
    // that lies within the boundaries
    
    fold_start = std::max (pulse_boundaries[subint_count], obs.start_time);

    if (verbose)
      cerr << "rawprofile::bound new subint starts at" 
	   << "\n max start=" << fold_start
	   << "\n cur start=" << pulse_boundaries[subint_count]
	   << "\n obs start=" << obs.start_time
	   << endl;
    
    for (; subint_count < pulse_boundaries.size()-1; subint_count++)
      if (pulse_boundaries[subint_count+1] > fold_start)
	break;

    if (subint_count >= pulse_boundaries.size()-1) {
      if (verbose)
	cerr << "rawprofile::bound observation start time past boundary"
	     << "\n bound start=" << pulse_boundaries.front()
	     << "\n bound end  =" << pulse_boundaries.back()
	     << endl;

      MJD::precision = oldMJDprecision;
      return false;
    }

  }

  if (verbose)
    cerr << "rawprofile::bound subint #" << subint_count << endl;
  
  // determine the amount of data to be integrated, and how far into
  // 'obs' to start
  MJD offset = fold_start - obs.start_time;
  istart = (unsigned) rint (offset.in_seconds() * obs.rate);

  if (verbose)
    cerr << "rawprofile::bound offset " << offset.in_seconds()
	 << "s (" << istart << "pts)" << endl;
  
  MJD fold_end = std::min (obs.end_time, pulse_boundaries[subint_count+1]);
  MJD fold_total = fold_end - fold_start;
  fndat = (unsigned) rint (fold_total.in_seconds() * obs.rate);

  if (verbose)
    cerr << "rawprofile::bound fold " << fold_total.in_seconds()
	 << "s (" << fndat << "pts) until"
         << "\n min end=" << fold_end
         << "\n cur end=" << pulse_boundaries[subint_count+1]
         << "\n obs end=" << obs.end_time
         << endl;

  if (fold_total.in_seconds() < 0.0) {
    // the current data end before the start subint of interest
    if (verbose)
      cerr << "rawprofile::bound data end before start of current subint=" 
	   << fold_start << endl;

    // return 0 - no error, but the user should check 
    // int_time before using the subint
    MJD::precision = oldMJDprecision;
    return false;
  }
  
  double actual = double(fndat)/obs.rate;
  if (verbose)
    cerr << "rawprofile::bound fold " << actual
	 << "s (" << fndat << " pts) out of "
	 << (obs.end_time - obs.start_time).in_seconds()
	 << "s (" << obs.ndat << " pts)." << endl;
  
  double samples_to_end = 
    (pulse_boundaries[subint_count+1] - fold_end).in_seconds() * obs.rate;

  if (verbose)
    cerr << "rawprofile::bound " << samples_to_end << " samples to"
      " end of current profile" << endl;

  if (samples_to_end < 0.5)
    profile_complete = true;

  if (fold_end < obs.end_time) {
    // the obs extends more than one subint, increment the
    // subint counter and set obs_extends_next so that the user 
    // will re-call this function to get the rest of the data
    subint_count ++;
    obs_extends_next = true;

    if (verbose) {
      if (subint_count < pulse_boundaries.size()-1)
	cerr << "rawprofile::bound observation extends next subint #" 
	     << subint_count << endl;
      else
	cerr << "rawprofile::bound end of sub-ints" << endl;
    }
  }
  
  if (istart + fndat > obs.ndat) {
    cerr << "rawprofile::bound INTERNAL ERROR *** correcting." << endl;
    fndat = obs.ndat - istart;
  }

  MJD::precision = oldMJDprecision;
  return true;
}


int rawprofile::mixable (const observation& obs, int nbin,
			 Int64 istart, Int64 fndat)
{
  MJD obsStart = obs.start_time + double (istart) / obs.rate;

  if (verbose)
    cerr << "rawprofile::mixable"
         << "\n obs start=" << obs.start_time 
	 << "\n cur start=" << obsStart << endl;

  MJD obsEnd;
  // if fndat not specified, will fold to the end of the observation
  // (works also for special case of adding rawprofiles together;
  // where using ndat=nbin would not make sense)
  if (fndat == 0)
    obsEnd = obs.end_time;
  else
    obsEnd = obsStart + double (fndat) / obs.rate;

  if (int_time > 0.0) {
    if (!combinable (obs)) {
      fprintf (stderr,
	       "rawprofile::mixable Cannot add differing observations\n");
      return -1;
    }
    if (ndat != nbin) {
      fprintf (stderr,
	       "rawprofile::mixable Cannot add profiles with diff ndat\n");
      return -1;
    }
    end_time = std::max (end_time, obsEnd);
    start_time = std::min (start_time, obsStart);

    if (verbose)
      cerr << "rawprofile::mixable combine start=" << start_time
	   << " end=" << end_time << endl;
  }
  else {
    // first time called
    observation::operator = (obs);
    ndat = nbin;
    end_time = obsEnd;
    start_time = obsStart;
    size_dataspace ();
    zero ();
  }
  return 0;
}

rawprofile::rawprofile () {
  init ();
}

rawprofile::rawprofile (int np_init, int nb_init)
{
  init();
  npol = np_init;
  ndat = nb_init;
  size_dataspace ();
  zero ();
}

rawprofile::~rawprofile()
{
  if (destroy_trace) {
    fprintf (stderr, "rawprofile: destructor entered.\n");
    fflush (stderr);
  }

  if (buffer) delete [] buffer;
  if (hits) delete [] hits;
  init();

  if (destroy_trace) {
    fprintf (stderr, "rawprofile: destructor exits.\n");
    fflush (stderr);
  }
}

rawprofile::rawprofile (const rawprofile & r)
{
  init();
  rawprofile::operator= (r);
}

rawprofile& rawprofile::operator = (const rawprofile & prof)
{
  if (this == &prof)
    return *this;

  observation::operator= (prof);
  ndat     = prof.ndat;
  int_time = prof.int_time;
  midtime  = prof.midtime;
  pfold    = prof.pfold;

  size_dataspace ();

  unsigned tpts = ndat * npol * nchan;
  float* tptr = buffer;
  float* fptr = prof.buffer;

  for (unsigned ipt=0; ipt<tpts; ipt++) {
    *tptr = *fptr;
    tptr++;
    fptr++;
  }

  for (int i=0; i<ndat; i++)
    hits[i] = prof.hits[i];

  return *this;
}

rawprofile & rawprofile::operator += (const rawprofile & prof)
{
  if (mixable (prof, prof.ndat) < 0) {
    fprintf (stderr,
	     "rawprofile::operator+= : mixable failed\n");
    return *this;
  }

  unsigned tpts = ndat * npol * nchan;
  float* tptr = buffer;
  float* fptr = prof.buffer;

  for (unsigned ipt=0; ipt<tpts; ipt++) {
    *tptr += *fptr;
    tptr++;
    fptr++;
  }

  for (int ibin=0; ibin<ndat; ibin++)
    hits[ibin] += prof.hits[ibin];

  int_time += prof.int_time;

  return *this;
}

rawprofile & rawprofile::operator *= (double norm)
{
  unsigned tpts = ndat * npol * nchan;
  float* tptr = buffer;

  for (unsigned ipt=0; ipt<tpts; ipt++) {
    *tptr *= norm;
    tptr++;
  }

  return *this;
}

const rawprofile operator + (rawprofile prof1, const rawprofile& prof2)
{
  return prof1 += prof2;
}

rawprofile::rawprofile (int nb, const polyco& polly, const float_Stream& sgnl)
{
  init();
  fold (nb, polly, sgnl);
}

rawprofile::rawprofile (int nb, double calperiod, const float_Stream& sgnl)
{
  init();
  fold (nb, calperiod, sgnl);
}

int rawprofile::fold_common (int nb, const float_Stream& sgnl, 
			     const polyco* polly, double calperiod,
			     Int64 istart, Int64 fndat)
{
  if (timestuff)
    time_fold.start();

  if (fndat == 0)
    fndat = sgnl.ndat;

  if (mixable (sgnl, nb, istart, fndat) < 0)
    return -1;


  const polynomial* tempo_polynomial = NULL;
  if (polly)
    tempo_polynomial = polly->nearest (sgnl.start_time);

  float* data[4];
  for (int i=0; i<4; i++)
    data[i] = sgnl.data[i];

  int incr=0;
  sgnl.datptr (0,0,incr);

  folding::fold (npol, nchan, ndat, buffer, hits,
		 data, sgnl.ch_bsize, incr, istart, fndat, 
		 sgnl.start_time, sgnl.rate,
		 sgnl.ppweight, sgnl.weights[0],
		 tempo_polynomial, calperiod,
		 &int_time);

  if (timestuff)
    time_fold.stop();

  return 0;
}



