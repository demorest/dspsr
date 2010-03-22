/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LevelMonitor.h"
#include "dsp/LevelHistory.h"

#include "dsp/IOManager.h"
#include "dsp/HistUnpacker.h"
#include "dsp/Input.h"

#include "dsp/WeightedTimeSeries.h"
#include "fsleep.h"

#include <assert.h>

using namespace std;

bool dsp::LevelMonitor::verbose = false;

//! Initialize null values
dsp::LevelMonitor::LevelMonitor ()
{
  abort = false;
  input = 0;
  history = 0;

  swap_polarizations = false;
  consecutive = true;

  n_integrate = 1 << 26;
  block_size = 1 << 18;

  mean_tolerance = var_tolerance = 0.01;
  max_iterations = 0;
  between_iterations = 1.0;

  far_from_good = false;

  setting_thresholds = true;
  stop_after_good = false;

  data = new WeightedTimeSeries;
}

dsp::LevelMonitor::~LevelMonitor ()
{
}

//! Set the device to be used to plot/log the digitizer statistics
void dsp::LevelMonitor::set_history (LevelHistory* _history)
{
  history = _history;

  if (unpacker)
    history->set_unpacker( unpacker );
}

//! Set the device to be used to plot/log the digitizer statistics
void dsp::LevelMonitor::set_input (IOManager* _input)
{
  input = _input;

  if (!input)
    return;

  if (data)
    input->set_output (data);

  input->prepare();

  input->set_block_size (block_size);

  Observation* info = input->get_info();

  nchan = info->get_nchan();
  npol = info->get_npol();
  ndim = info->get_ndim();

  unpacker = dynamic_cast<HistUnpacker*>(input->get_unpacker());

  if (!unpacker)
    throw Error (InvalidState, "dsp::LevelMonitor::set_input",
                 "unpacker is not a HistUnpacker; optimal_variance unknown");

  if (history)
    history->set_unpacker( unpacker );

  if (unpacker->get_ndim_per_digitizer() == 2)
    ndim = 1;

  optimal_variance = unpacker->get_optimal_variance();

  cerr << "dsp::LevelMonitor optimal_variance=" << optimal_variance << endl;

  ndig = nchan * npol * ndim;
}

//! Set the number of points included in each calculation of thresholds
void dsp::LevelMonitor::set_integration (uint64_t npts)
{
  n_integrate = npts;
}

void dsp::LevelMonitor::set_between_iterations (double seconds)
{
  between_iterations = seconds;
}

//! Abort monitoring
void dsp::LevelMonitor::monitor_abort ()
{
  abort = true;
}

/*!
  This method should be redefined by sub-classes which implement actual
  control over a physical/virtual digitizer
*/
int dsp::LevelMonitor::change_gain (unsigned ichan,
				    unsigned ipol,
				    unsigned idim,
				    double delta_gain)
{
  cout << "GAIN " 
       << ichan << " " 
       << ipol << " " 
       << idim << " " 
       << delta_gain << endl;

  return 0;
}

/*!
  This method should be redefined by sub-classes which implement actual
  control over a physical/virtual digitizer
*/
int dsp::LevelMonitor::change_levels (unsigned ichan,
				      unsigned ipol,
				      unsigned idim,
				      double delta_mean)
{
  cout << "LEVEL " 
       << ichan << " "
       << ipol << " " 
       << idim << " "
       << delta_mean << endl;

  return 0;
}

void dsp::LevelMonitor::set_max_iterations (unsigned iter)
{
  max_iterations = iter;
}

void dsp::LevelMonitor::set_swap_polarizations (bool swap)
{
  swap_polarizations = swap;
}

void dsp::LevelMonitor::set_consecutive (bool flag)
{
  consecutive = flag;
}

/*! 
  Monitor the data coming from the Bit_Stream_Input and issue appropriate
  level-setting commands
*/
void dsp::LevelMonitor::monitor ()
{
  vector<double> mean;
  vector<double> variance;

  unsigned iterations = 0;

  while ( (!max_iterations || iterations < max_iterations) &&
          !input->get_input()->eod() )
  {
    if (verbose)
      cerr << "LevelMonitor::monitor accumulate stats ..." << endl;

    if (accumulate_stats (mean, variance) < 0)
    {
      cerr << "LevelMonitor::monitor fail accumulate_stats" << endl;
      return;
    }

    if (setting_thresholds)
    {
      if (verbose)
        cerr << "LevelMonitor::monitor call set thresholds ..." << endl;

      if (set_thresholds (mean, variance) < 0)
      {
        cerr << "LevelMonitor::monitor fail set_thresholds" << endl;
        return;
      }
    }

    if (history)
    {
      if (verbose)
	cerr << "LevelMonitor::monitor log statistics ..." << endl;

      history -> log_stats (mean, variance);
    }
    
    if (!far_from_good)
    {
      if (verbose)
        cerr << "LevelMonitor::monitor not far from good; returning" << endl;
      return;
    }

    if (between_iterations)
    {
      if (verbose)
        cerr << "LevelMonitor::monitor sleeping " << between_iterations << " seconds" << endl;
      fsleep (between_iterations);
    }

    iterations++;

  }
}

int dsp::LevelMonitor::accumulate_stats (vector<double>& mean,
					 vector<double>& variance) try
{
  if (!input)
  {
    cerr << "LevelMonitor::accumulate_stats no data input" << endl;;
    return -1;
  }

  if (verbose)
    cerr << "LevelMonitor::accumulate_stats integrate " << n_integrate << endl;

  // get only the latest information from the digitizer
  if (!consecutive)
  {
    if (verbose)
      cerr << "LevelMonitor::accumulate_stats SEEK_END" << endl;
    input -> get_input() -> seek (0, SEEK_END);
  }

  if (verbose)
    cerr << "LevelMonitor::accumulate_stats finished seek" << endl;

  unsigned input_ndim = input->get_info()->get_ndim();

  // start with an empty histogram ...
  if (unpacker)
    unpacker -> zero_histogram ();

  // ... and empty sums
  vector<double> tot_sum (ndig, 0.0);
  vector<double> tot_sumsq (ndig, 0.0);
  vector<uint64_t> tot_pts (ndig, 0);

  while (!abort && tot_pts[0] < n_integrate && !input->get_input()->eod())
  {
    input -> load (data);

    uint64_t ndat = data->get_ndat();
    unsigned ppweight = data->get_ndat_per_weight ();
    uint64_t nweight = ndat / ppweight;

    // combine the statistics for real and imaginary components
    if (input_ndim == 2 && ndim == 1)
      ppweight *= 2;

    if (verbose)
      cerr << "LevelMonitor::accumulate_stats loaded ndat=" << ndat << endl;

    unsigned idig = 0;

    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
	for (unsigned idim=0; idim < ndim; idim++)
	{
	  float* dat = data->get_datptr (ichan, ipol) + idim;
	  unsigned* wt = data->get_weights (ichan, ipol);

	  for (unsigned iwt=0; iwt < nweight; iwt++)
	  {
	    if (wt[iwt] == 0)
	      continue;

            double sum = 0;
            double sumsq = 0;

	    for (unsigned i=0; i<ppweight; i++)
	    {
	      float val = dat[ iwt*ppweight + i*ndim ];
	      sum += val;
	      sumsq += val*val;
	    }

	    tot_sum[idig] += sum;
	    tot_sumsq[idig] += sumsq;
	    tot_pts[idig] += ppweight;
	  }

	  idig ++;
	}
      }
    }

    assert (idig == ndig);
  }

  mean.resize (ndig);
  variance.resize (ndig);

  for (unsigned idig=0; idig < ndig; idig++) 
  {
    if (tot_pts[idig] == 0)
    {
      mean[idig] = variance[idig] = 0.0;
      continue;
    }

    double x = tot_sum[idig] / tot_pts[idig];
    double xsq = tot_sumsq[idig] / tot_pts[idig];

    mean[idig] = x;
    variance[idig] = xsq - x*x;
  }

  return 0;
}
catch (Error& error)
{
  throw error += "dsp::LevelMonitor::accumulate_stats";
}

int dsp::LevelMonitor::set_thresholds (vector<double>& mean,
				       vector<double>& variance)
{
  if (mean.size() != ndig || variance.size() != ndig)
  {
    cerr << "LevelMonitor::set_thresholds size mismatch" << endl;
    return -1;
  }

  far_from_good = false;

  if (verbose)
    cerr << "LevelMonitor::set_thresholds ndig=" << ndig << endl;

  unsigned idig = 0;

  for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      for (unsigned idim=0; idim < ndim; idim++)
      {
        if (variance[idig] <= 0.0)
        {
          idig ++;
          continue;
        }

        double delta_var = fabs(variance[idig] - optimal_variance);

        if (verbose)
          cerr << "LevelMonitor::set_thresholds idig=" << idig
	       << " dvar=" << delta_var << " max=" << var_tolerance << endl;

        if (delta_var < var_tolerance)
        {
          if (stop_after_good)
          {
            if (verbose)
              cerr << "LevelMonitor::set_thresholds stopping on good" << endl;
            setting_thresholds = false;
          }
        }
        else
        {   
          // don't bother adjusting the trim while the gain is improperly set
          if (delta_var > 5 * var_tolerance)
          {
	    if (verbose)
	      cerr << "LevelMonitor::set_thresholds hold the trim" << endl;
	    far_from_good = true;
          }

          // calculate the change in gain
          double delta_gain = sqrt(optimal_variance / variance[idig]);

          if (!isfinite(delta_gain))
          {
            idig ++;
            continue;
          }

          if (verbose)
	    cerr << "LevelMonitor::set_thresholds change_gain (" 
	         << idig << ", " << delta_gain << ")" << endl;
    
          if ( change_gain (ichan, ipol, idim, delta_gain) < 0 )
          {
	    cerr << "LevelMonitor::set_thresholds fail change_gain\n";
	    return -1;
          }
          if (verbose)
	    cerr << "LevelMonitor::set_thresholds change gain complete" << endl;
        }
        
        if (!far_from_good && (fabs(mean[idig]) > mean_tolerance))
        {
          if (verbose) cerr << "LevelMonitor::set_thresholds adjust trim\n";
    
          if ( change_levels (ichan, ipol, idim, mean[idig]) < 0 )
          {
	    cerr << "LevelMonitor::set_thresholds fail change_level\n";
	    return -1;
          }
    
          if (verbose)
	    cerr << "LevelMonitor::set_thresholds change level complete" << endl;
        }

        idig ++;
      }
    }
  }

  return 0;
}

