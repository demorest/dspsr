/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ExcisionStatsPlotter.h"
#include "dsp/ExcisionUnpacker.h"
#include "dsp/BitSeries.h"
#include "dsp/TimeSeries.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "Error.h"

#include "strutil.h"
#include "dirutil.h"
#include "templates.h"

#include <cpgplot.h>

#include <iostream>
#include <unistd.h>

using namespace std;

static char* args = "c:D:hn:S:s:T:t:vVw:";

void usage ()
{
  cout << "digistat - plots digitizer statistics\n"
    "Usage: digistat [" << args << "] file1 [file2 ...] \n"
    "Unpacking parameters:\n"
    " -n <nsample>   number of samples used in estimating undigitized power\n"
    " -c <cutoff>    cutoff threshold for impulsive interference excision\n"
    " -t <threshold> sampling threshold at record time\n"
    "Display paramters:\n"
    " -D <device>    set the pgplot device to use\n"
    " -s <seconds>   total amount of data in each plot\n"
    " -w <seconds>   amount of data averaged into each point in plot\n"
       << endl;
}

int main (int argc, char** argv) try 
{
  char* metafile = 0;
  bool display = true;
  bool verbose = false;

  // unknown until data are loaded
  unsigned excision_nsample = 0;

  // disable excision by default
  float excision_cutoff = 0.0;

  // unknown until data are loaded
  float excision_threshold = 0.0;

  float time_per_plot = 1.0;
  float time_per_point = 1e-3;

  double seek_seconds = 0;
  double total_seconds = 0;

  string pgdev = "?";

  int c;
  int scanned;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'c':
      scanned = sscanf (optarg, "%f", &excision_cutoff);
      if (scanned != 1) {
        cerr << "digistat: error parsing " << optarg << " as"
          " dynamic output level assignment cutoff" << endl;
        return -1;
      }
      break;

    case 'h':
      usage ();
      return 0;

    case 'n':
      scanned = sscanf (optarg, "%u", &excision_nsample);
      if (scanned != 1) {
	cerr << "digistat: error parsing " << optarg << " as"
	  " number of samples used to estimate undigitized power" << endl;
	return -1;
      }
      break;

    case 's':
      scanned = sscanf (optarg, "%f", &time_per_plot);
      if (scanned != 1) {
        cerr << "digistat: error parsing " << optarg << " as"
          " time per plot" << endl;
        return -1;
      }
      break;

    case 't':
      scanned = sscanf (optarg, "%f", &excision_threshold);
      if (scanned != 1) {
        cerr << "digistat: error parsing " << optarg << " as"
          " sampling threshold" << endl;
        return -1;
      }
      break;

    case 'S':
      seek_seconds = strtod (optarg, 0);
      break;

    case 'T':
      total_seconds = strtod (optarg, 0);
      break;

    case 'V':
      dsp::Operation::verbose = true;
      dsp::Observation::verbose = true;
    case 'v':
      verbose = true;
      break;

    case 'w':
      scanned = sscanf (optarg, "%f", &time_per_point);
      if (scanned != 1) {
        cerr << "digistat: error parsing -w " << optarg << endl;
        return -1;
      }
      cerr << "digistat: time per point=" << time_per_point << " s" << endl;
      break;

    case 'D':
      pgdev = optarg;
      if (pgdev.empty()) {
        pgdev = "?";
      }
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  if (time_per_point > 0.5 * time_per_plot) {
    cerr << "digistat: Cannot plot less than two points at a time\n"
      " -w <point> must be less than half of -s <plot>\n" << endl;
    return -1;
  }

  vector <string> filenames;

  if (metafile)
    stringfload (&filenames, metafile);
  else 
    for (int ai=optind; ai<argc; ai++)
      dirglob (&filenames, argv[ai]);

  if (filenames.size() == 0) {
    usage ();
    return 0;
  }

  if (display) {
    cpgbeg (0, pgdev.c_str(), 0, 0);
    cpgask(1);
    cpgsvp (0.05, 0.95, 0.0, 0.95);
    cpgsch (2.0);
  }

  // converted voltages container
  Reference::To<dsp::TimeSeries> voltages = new dsp::TimeSeries;

  // interface manages the creation of data loading and converting classes
  Reference::To<dsp::IOManager> manager = new dsp::IOManager;

  manager->set_output (voltages);

  // plots digitization statistics
  Reference::To<dsp::BitStatsPlotter> plotter;
  Reference::To<dsp::ExcisionUnpacker> excision;
  Reference::To<dsp::HistUnpacker> unpack;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try
  {
    cerr << "digistat: opening file " << filenames[ifile] << endl;
    manager->open (filenames[ifile]);

    if (verbose)
      cerr << "digistat: file " << filenames[ifile] << " opened" << endl;

    excision = dynamic_cast<dsp::ExcisionUnpacker*>(manager->get_unpacker());

    if (excision)
    {
      // plots two-bit digitization statistics
      plotter = new dsp::ExcisionStatsPlotter;
      plotter->set_data( excision );

      if ( excision_nsample )
	excision -> set_ndat_per_weight ( excision_nsample );
	
      if ( excision_threshold )
	excision -> set_threshold ( excision_threshold );

      cerr << "digistat: setting excision unpacker cutoff to "
           << excision_cutoff << endl;
 
      excision -> set_cutoff_sigma ( excision_cutoff );

      unpack = excision;
    }
    else
    {
      unpack = dynamic_cast<dsp::HistUnpacker*>(manager->get_unpacker());
      
      if (unpack)
      {
	plotter = new dsp::BitStatsPlotter;
	plotter->set_data( unpack );
      }
      else
      {
	cerr << "digistat: Unpacker does not maintain a histogram" << endl;
	plotter = 0;
      }
    }

    if (seek_seconds)
      manager->get_input()->seek_seconds (seek_seconds);
    
    if (total_seconds)
      manager->get_input()->set_total_seconds (seek_seconds + total_seconds);

    cerr << "Bandwidth = " << manager->get_info()->get_bandwidth() << endl;
    cerr << "Sampling rate = " << manager->get_info()->get_rate() << endl;

    if (manager->get_info()->get_rate() <= 0.0)
    {
      cerr << "digistat: invalid sampling rate" << endl;
      return -1;
    }

    // set the number of samples to load
    double samples = manager->get_info()->get_rate() * time_per_plot;
    uint64_t block_size = uint64_t(samples + 0.5);
    time_per_plot = double(block_size) / manager->get_info()->get_rate();

    cerr << block_size << " samples per " 
         << time_per_plot << "s plot" << endl;

    uint64_t ram = manager->set_block_size( block_size );
    double megabyte = 1024*1024;
    cerr << "digistat: blocksize=" << manager->get_input()->get_block_size()
         << " samples or " << double(ram)/megabyte << " MB" << endl;

    // set the number of samples to average
    samples = manager->get_info()->get_rate() * time_per_point;
    uint64_t point_size = uint64_t(samples + 0.5);
    time_per_point = double(point_size) / manager->get_info()->get_rate();

    cerr << point_size << " samples per " 
         << time_per_point*1e6 << "us point" << endl;

    uint64_t npoints = block_size / point_size;
    if (block_size % point_size)
      npoints ++;

    cerr << npoints << " points per plot" << endl;

    vector<float> xaxis (npoints);
    vector<float> mean  (npoints);
    vector<float> rms   (npoints);

    double current_time = 0.0;

    while (!manager->get_input()->eod())
    {
      if (unpack)
        unpack->zero_histogram ();
      
      manager->load (voltages);

      if (verbose)
        cerr << "input sample=" << voltages->get_input_sample() 
             << " ndat=" << voltages->get_ndat() << endl;

      if (voltages->get_ndat() < point_size)
        break;

      for (unsigned ichan=0; ichan<voltages->get_nchan(); ++ichan)
      {
	if (display)
	  cpgpage();
	
	float bottom = 0.52;

	for (unsigned ipol=0; ipol<voltages->get_npol(); ++ipol)
        {
	  if (display && plotter) try
          {
	    cpgsvp (0.7, 0.95, bottom, bottom+0.40);
	    plotter->plot (ichan,ipol);
	  }
    catch (Error& e) { if (verbose) cerr << e << endl; }

    cpgsch (1.0);
    cerr << "ichan=" << ichan << " ipol=" << ipol << endl;

	  float* data = voltages->get_datptr (ichan, ipol);
          uint64_t nfloat = voltages->get_ndat() * voltages->get_ndim();
          uint64_t npoint = point_size * voltages->get_ndim();
	  uint64_t idat = 0;

	  for (unsigned ipt=0; ipt<npoints; ipt++)
          {
	    xaxis[ipt] = current_time + double(ipt)*time_per_point;

            double sum = 0, sumsq = 0;
	    unsigned count = 0;
	    
	    for (uint64_t jdat=0; jdat<npoint && idat<nfloat; jdat++)
            {
	      float sample = data[idat]; idat ++;

	      sum += sample;
	      sumsq += sample*sample;
	      count ++;
	    }

            if (count == 0)
              count = 1;
               
	    mean[ipt] = sum/count;
	    double var  = (sumsq - sum*sum/count) / count;

	    rms[ipt] = 0.0;
            if (var > 0)
              rms[ipt] = sqrt(var);
	  }

	  float min = 0.0;
	  float max = 0.0;
	  float buf = 0.0;

	  // plot the mean
	  
	  minmax (mean, min, max);
	  buf = (max-min) * 0.05;

          // cerr << "mean min=" << min << " max=" << max << endl;

          if (min == max)
            buf = 1.0;
 
	  cpgswin (current_time, current_time+time_per_plot, min-buf, max+buf);
	  cpgsvp (0.1, 0.63, bottom, bottom+0.20);
	  cpgsci(1);
	  
	  if (ipol==1)
          {
	    cpglab("Seconds from start of file", "", "");
	    cpgbox("bcnst",0.0,0,"bcnvst",0.0,0);
	  }
	  else
	    cpgbox("bcst",0.0,0,"bcnvst",0.0,0);
	  cpgmtxt("L",3.5,.5,.5,"mean");
	  
	  cpgsci(5);
	  cpgpt(npoints, &(xaxis[0]), &(mean[0]), -1);
	  
	  // plot the rms
	  
	  minmax (rms, min, max);
	  buf = (max-min) * 0.05;

          // cerr << "rms min=" << min << " max=" << max << endl;

          if (min == max)
            buf = 1.0;

	  cpgswin (current_time, current_time+time_per_plot, min-buf, max+buf);
	  cpgsvp (0.1, 0.63, bottom+0.20, bottom+0.40);
	  cpgsci(1);
	  cpgbox("bcst",0.0,0,"bcnvst",0.0,0);
	  cpgmtxt("L",3.5,.5,.5,"rms");
	  
	  cpgsci(6);
	  cpgpt(npoints, &(xaxis[0]), &(rms[0]), -1);
	  
	  bottom = 0.08;
	}
	
	if (display && plotter) 
        {
	  cpgsvp (0.7, 0.95, bottom, 0.92);
	  plotter->label ();
	}
      }

      current_time += time_per_plot;
    }

    if (verbose)
      cerr << "end of data file " << filenames[ifile] << endl;
  }
  catch (string& error) {
    cerr << error << endl;
  }
  
  if (display)
    cpgend();
  
  return 0;
}

catch (Error& error)
{
  cerr << error << endl;
  return -1;
}

catch (string& error)
{
  cerr << "exception thrown: " << error << endl;
  return -1;
}

catch (...)
{
  cerr << "exception thrown: " << endl;
  return -1;
}

