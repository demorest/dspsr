/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LevelMonitor.h"
#include "dsp/LevelHistory.h"

#include "dsp/IOManager.h"
#include "dsp/HistUnpacker.h"
#include "dsp/Input.h"

#include "dsp/TimeSeries.h"

#include <assert.h>

using namespace std;

bool dsp::LevelMonitor::verbose = false;
bool dsp::LevelMonitor::connect = true;

//! Initialize null values
dsp::LevelMonitor::LevelMonitor ()
{
  abort = false;
  connect = true;
  input = 0;
  history = 0;

  n_integrate = (unsigned long) 64e6;
  mean_tolerance = var_tolerance = 5e-4;

  far_from_good = false;

  data = new TimeSeries;
}

dsp::LevelMonitor::~LevelMonitor ()
{
}

//! Set the device to be used to plot/log the digitizer statistics
void dsp::LevelMonitor::set_history (LevelHistory* _history)
{
  history = _history;
}

//! Set the device to be used to plot/log the digitizer statistics
void dsp::LevelMonitor::set_input (IOManager* _input)
{
  input = _input;

  if (input)
    unpacker = dynamic_cast<HistUnpacker*>(input->get_unpacker());
}

//! Set the number of points included in each calculation of thresholds
void dsp::LevelMonitor::set_integration (uint64 npts)
{
  n_integrate = npts;
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
int dsp::LevelMonitor::change_gain (int channel, double delta_dBm)
{
  cerr << "LevelMonitor::change_gain "
       << channel << " " << delta_dBm << endl;
  return 0;
}

/*!
  This method should be redefined by sub-classes which implement actual
  control over a physical/virtual digitizer
*/
int dsp::LevelMonitor::change_levels (int channel, double delta_Volt)
{
  cerr << "LevelMonitor::change_levels "
       << channel << " " << delta_Volt << endl;
  return 0;
}

//! Convert power to dBm
double variance2dBm (double variance)
{
  return 10.0 * log (variance) / log(10.0);
}

/*! 
  Monitor the data coming from the Bit_Stream_Input and issue appropriate
  level-setting commands
*/
void dsp::LevelMonitor::monitor ()
{
  vector<double> mean;
  vector<double> variance;

  int bad_level_count = 0;

  while (!abort)
  {
    if (verbose)
      cerr << "LevelMonitor::monitor accumulate stats ..." << endl;

    if (accumulate_stats (mean, variance) < 0)
    {
      cerr << "LevelMonitor::monitor fail accumulate_stats" << endl;
      return;
    }

    if (verbose)
      cerr << "LevelMonitor::monitor call set thresholds ..." << endl;

    if (set_thresholds (mean, variance) < 0)
    {
      cerr << "LevelMonitor::monitor fail set_thresholds" << endl;
      return;
    }

    if (history)
    {
      if (verbose)
	cerr << "LevelMonitor::monitor log statistics ..." << endl;

      history -> log_stats (mean, variance, unpacker);
    }
    
    if (!far_from_good)
    {
      bad_level_count = 0;
      rest_a_while();
    }
    else
    {
      bad_level_count++;

      if (bad_level_count > iterations && iterations != 0 )
      {
         cerr << "****\nLevelMonitor::Cannot set levels: level count greater than " << iterations << "\n****" << endl; 
         // monitor_abort();   
         bad_level_count = 0;
         cerr << "****\nLevelMonitor::Resetting iteration count and sleeping " << "\n****" << endl;
         sleep(2); 
      }
    }    
  }
}

int dsp::LevelMonitor::accumulate_stats (vector<double>& mean,
					 vector<double>& variance)
{
  if (!input)
  {
    cerr << "LevelMonitor::accumulate_stats no data input" << endl;;
    return -1;
  }

  if (verbose)
    cerr << "LevelMonitor::accumulate_stats integrate " << n_integrate << endl;

  // get only the latest information from the digitizer
  input -> get_input() -> seek (0, SEEK_END);

  if (verbose)
    cerr << "LevelMonitor::accumulate_stats finished seek" << endl;

  Observation* info = input->get_info();

  /*
    This function computes the statistics for what are in principle
    independent data streams.  The interpretation of these statistics
    in terms of the actual number of indepdendent digitizer control
    channels is done later
  */

  unsigned nchan = info->get_nchan();
  unsigned npol = info->get_npol();
  unsigned ndim = info->get_ndim();
  unsigned ndig = nchan * npol * ndim;

  // start with an empty histogram ...
  if (unpacker)
    unpacker -> zero_histogram ();

  // ... and empty sums
  vector<double> tot_sum (ndig, 0.0);
  vector<double> tot_sumsq (ndig, 0.0);

  uint64 total_pts = 0;

  while (!abort && total_pts < n_integrate)
  {
    input -> load (data);

    if (verbose)
      cerr << "LevelMonitor::accumulate_stats data loaded" << endl;

    uint64 ndat = data->get_ndat();
    unsigned idig = 0;

    for (unsigned ichan=0; ichan < nchan; ichan++)
      for (unsigned ipol=0; ipol < npol; ipol++)
	for (unsigned idim=0; idim < ndim; idim++)
	{
	  float* ptr = data->get_datptr (ichan, ipol) + idim;
	  double sum = 0;
	  double sumsq = 0;
	  for (unsigned i=0; i<ndat; i++)
	  {
	    float val = ptr[i*ndim];
	    sum += val;
	    sumsq += val*val;
	  }
	  tot_sum[idig] += sum;
	  tot_sumsq[idig] += sumsq;
	  idig ++;
	}

    total_pts += ndat;
    
    assert (idig == ndig);

  }

  mean.resize (ndig);
  variance.resize (ndig);

  for (unsigned idig=0; idig < ndig; idig++) 
  {
    double x = tot_sum[idig] / total_pts;
    double xsq = tot_sumsq[idig] / total_pts;

    mean[idig] = x;
    variance[idig] = xsq - x*x;
  }

  return 0;
}

int dsp::LevelMonitor::set_thresholds (vector<double>& mean,
					vector<double>& variance)
{
  unsigned ndig = mean.size();
  if (mean.size() != variance.size()) {
    cerr << "LevelMonitor::set_thresholds size mismatch" << endl;
    return -1;
  }

  far_from_good = false;

  for (unsigned idig=0; idig < ndig; idig ++) {
    
    double delta_var = fabs(variance[idig] - optimal_variance);

    if ((delta_var > var_tolerance) && (connect == true)) {
      
      // don't bother adjusting the trim while the gain is improperly set
      if (delta_var > 5 * var_tolerance) {
	if (verbose)
	  cerr << "LevelMonitor::set_thresholds hold the trim" << endl;
	far_from_good = true;
      }
      
      // calculate the change in gain
      double delta_dBm = optimal_dBm - variance2dBm (variance[idig]);
      if (verbose)
	cerr << "LevelMonitor::set_thresholds change_gain (" 
	     << idig << ", " << delta_dBm << ")" << endl;

      if ( change_gain (idig, delta_dBm) < 0 ) {
	fprintf (stderr, "LevelMonitor::set_thresholds fail change_gain\n");
	return -1;
      }
      if (verbose)
	cerr << "LevelMonitor::set_thresholds change gain complete" << endl;
    }
    
    if ((!far_from_good) && (fabs(mean[idig]) > mean_tolerance) && (connect == true)) {

      if (verbose) fprintf (stderr, "LevelMonitor::set_thresholds adjust trim\n");

      if ( change_levels (idig, mean[idig]) < 0 ) {
	fprintf (stderr, "LevelMonitor::set_thresholds fail change_level\n");
	return -1;
      }

      if (verbose)
	cerr << "LevelMonitor::set_thresholds change level complete" << endl;
    }
  }
  return 0;
}

