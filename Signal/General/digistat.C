/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/TwoBitStatsPlotter.h"
#include "dsp/TwoBitCorrection.h"
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

static char* args = "c:hn:s:t:vVw:";

void usage ()
{
  cout << "digistat - plots digitizer statistics\n"
    "Usage: digistat [" << args << "] file1 [file2 ...] \n"
    "Unpacking parameters:\n"
    " -n <nsample>   number of samples used in estimating undigitized power\n"
    " -c <cutoff>    cutoff threshold for impulsive interference excision\n"
    " -t <threshold> sampling threshold at record time\n"
    "Display paramters:\n"
    " -s <seconds>   total amount of data in each plot\n"
    " -w <seconds>   amount of data averaged into each point in plot\n"
       << endl;
}

int main (int argc, char** argv) try 
{
  char* metafile = 0;
  bool display = true;
  bool verbose = false;

  unsigned tbc_nsample = 0;
  float tbc_cutoff = 0.0;
  float tbc_threshold = 0.0;

  float time_per_plot = 1.0;
  float time_per_point = 1e-3;

  int c;
  int scanned;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'c':
      scanned = sscanf (optarg, "%f", &tbc_cutoff);
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
      scanned = sscanf (optarg, "%u", &tbc_nsample);
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
      scanned = sscanf (optarg, "%f", &tbc_threshold);
      if (scanned != 1) {
        cerr << "digistat: error parsing " << optarg << " as"
          " sampling threshold" << endl;
        return -1;
      }
      break;

    case 'V':
      dsp::Operation::verbose = true;
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
    cpgbeg (0, "?", 0, 0);
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
  Reference::To<dsp::TwoBitCorrection> correct;
  Reference::To<dsp::HistUnpacker> unpack;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try {

    cerr << "digistat: opening file " << filenames[ifile] << endl;
    manager->open (filenames[ifile]);

    if (verbose)
      cerr << "digistat: file " << filenames[ifile] << " opened" << endl;

    correct = dynamic_cast<dsp::TwoBitCorrection*>(manager->get_unpacker());

    if (correct) {

	// plots two-bit digitization statistics
	plotter = new dsp::TwoBitStatsPlotter;
	plotter->set_data( correct );

	if ( tbc_nsample )
	    correct -> set_nsample ( tbc_nsample );
	
	if ( tbc_threshold )
	    correct -> set_threshold ( tbc_threshold );
	
	if ( tbc_cutoff )
	    correct -> set_cutoff_sigma ( tbc_cutoff );

        unpack = correct;

    }

    else {

	unpack = dynamic_cast<dsp::HistUnpacker*>(manager->get_unpacker());
	
	if (unpack) {
	    plotter = new dsp::BitStatsPlotter;
	    plotter->set_data( unpack );
	}
	else {
	    cerr << "digistat: Unpacker does not maintain a histogram" << endl;
	    plotter = 0;
        }

    }

    cerr << "Bandwidth = " << manager->get_info()->get_bandwidth() << endl;
    cerr << "Sampling rate = " << manager->get_info()->get_rate() << endl;

    if (manager->get_info()->get_rate() <= 0.0)  {
      cerr << "digistat: invalid sampling rate" << endl;
      return -1;
    }

    // set the number of samples to load
    double samples = manager->get_info()->get_rate() * time_per_plot + 0.5;
    uint64 block_size = uint64(samples);
    time_per_plot = double(block_size) / manager->get_info()->get_rate();

    cerr << block_size << " samples per " << time_per_plot << "s plot" << endl;

    manager->get_input()->set_block_size( block_size );

    // set the number of samples to average
    samples = manager->get_info()->get_rate() * time_per_point + 0.5;
    uint64 point_size = uint64(samples);
    time_per_point = double(point_size) / manager->get_info()->get_rate();

    uint64 npoints = block_size / point_size;
    if (block_size % point_size)
      npoints ++;
    
    vector<float> xaxis (npoints);
    vector<float> mean  (npoints);
    vector<float> rms   (npoints);

    double current_time = 0.0;

    while (!manager->get_input()->eod()) {

      if (unpack)
        unpack->zero_histogram ();
      
      manager->load (voltages);

      for (unsigned ichan=0; ichan<voltages->get_nchan(); ++ichan) {

	if (display && plotter)
	  cpgpage();
	
	float bottom = 0.52;

	for (unsigned ipol=0; ipol<voltages->get_npol(); ++ipol) {

	  if (display && plotter) {
	    cpgsvp (0.7, 0.95, bottom, bottom+0.40);
	    plotter->plot (ichan,ipol);
	  }

	  float* data = voltages->get_datptr (ichan, ipol);
	  uint64 ndat = voltages->get_ndat();
	  uint64 idat = 0;

	  for (unsigned ipt=0; ipt<npoints; ipt++) {

	    xaxis[ipt] = current_time + double(ipt)*time_per_point;

	    mean[ipt] = 0;
	    rms [ipt] = 0;
	    unsigned count = 0;
	    
	    for (uint64 jdat=0; jdat<point_size && idat<ndat; jdat++)  {
	      float sample = data[idat]; idat ++;
	      mean[ipt] += sample;
	      rms [ipt] += sample*sample;
	    count ++;
	    }
	    
	    mean[ipt] /= count;
	    rms[ipt]  /= count;
	    rms[ipt]  = sqrt(rms[ipt] - mean[ipt]*mean[ipt]);
	    
	  }
	  
	  float min = 0.0;
	  float max = 0.0;
	  float buf = 0.0;

	  // plot the mean
	  
	  minmaxval (mean, min, max);
	  buf = (max-min) * 0.05;
	  
	  cpgswin (current_time, current_time+time_per_plot, min-buf, max+buf);
	  cpgsvp (0.1, 0.63, bottom, bottom+0.20);
	  cpgsci(1);
	  
	  if (ipol==1) {
	    cpglab("Seconds from start of file", "", "");
	    cpgbox("bcnst",0.0,0,"bcnvst",0.0,0);
	  }
	  else
	    cpgbox("bcst",0.0,0,"bcnvst",0.0,0);
	  cpgmtxt("L",3.5,.5,.5,"mean");
	  
	  cpgsci(5);
	  cpgpt(npoints, &(xaxis[0]), &(mean[0]), -1);
	  
	  // plot the rms
	  
	  minmaxval (rms, min, max);
	  buf = (max-min) * 0.05;
	  
	  cpgswin (current_time, current_time+time_per_plot, min-buf, max+buf);
	  cpgsvp (0.1, 0.63, bottom+0.20, bottom+0.40);
	  cpgsci(1);
	  cpgbox("bcst",0.0,0,"bcnvst",0.0,0);
	  cpgmtxt("L",3.5,.5,.5,"rms");
	  
	  cpgsci(6);
	  cpgpt(npoints, &(xaxis[0]), &(rms[0]), -1);
	  
	  bottom = 0.08;

	}
	
	if (display && plotter) {
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

catch (Error& error) {
  cerr << error << endl;
  return -1;
}

catch (string& error) {
  cerr << "exception thrown: " << error << endl;
  return -1;
}

catch (...) {
  cerr << "exception thrown: " << endl;
  return -1;
}

