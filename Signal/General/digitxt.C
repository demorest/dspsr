/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TwoBitCorrection.h"
#include "dsp/BitSeries.h"
#include "dsp/TimeSeries.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "Error.h"

#include "strutil.h"
#include "dirutil.h"
#include "templates.h"

#include <iostream>
#include <unistd.h>

using namespace std;

static char* args = "ahM:vV";

void usage ()
{
  cout << "digitxt - print digitizer outputs on stdout\n"
    "Usage: digitxt file1 [file2 ...] \n" << endl;
}

int main (int argc, char** argv) try 
{
  char* metafile = 0;
  bool verbose = false;
  bool argand = false;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'a':
      argand = true;
      break;

    case 'h':
      usage ();
      return 0;

    case 'M':
      metafile = optarg;
      break;

    case 'V':
      dsp::Operation::verbose = true;
      dsp::Observation::verbose = true;
    case 'v':
      verbose = true;
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  vector <string> filenames;

  if (metafile)
    stringfload (&filenames, metafile);
  else 
    for (int ai=optind; ai<argc; ai++)
      dirglob (&filenames, argv[ai]);

  if (filenames.size() == 0)
  {
    usage ();
    return 0;
  }

  // converted voltages container
  Reference::To<dsp::TimeSeries> voltages = new dsp::TimeSeries;

  // interface manages the creation of data loading and converting classes
  Reference::To<dsp::IOManager> manager = new dsp::IOManager;

  manager->set_output (voltages);

  dsp::TwoBitCorrection* correct = 0;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try
  {
    cerr << "digitxt: opening file " << filenames[ifile] << endl;
    manager->open (filenames[ifile]);

    if (verbose)
      cerr << "digitxt: file " << filenames[ifile] << " opened" << endl;

    correct = dynamic_cast<dsp::TwoBitCorrection*>(manager->get_unpacker());

    if (correct)
      correct -> set_cutoff_sigma (0.0);

    if (verbose)
    {
      cerr << "Bandwidth = " << manager->get_info()->get_bandwidth() << endl;
      cerr << "Sampling rate = " << manager->get_info()->get_rate() << endl;
    }

    uint64_t block_size = 1024;

    if (verbose)
      cerr << "digitxt: set block_size=" << block_size << endl;

    manager->set_block_size( block_size );

    while (!manager->get_input()->eod())
    {
      manager->load (voltages);

      if (verbose)
        cerr << "input sample=" << voltages->get_input_sample() 
             << " ndat=" << voltages->get_ndat() << endl;

      const unsigned nchan = voltages->get_nchan();
      const unsigned npol = voltages->get_npol();
      const unsigned ndim = voltages->get_ndim();
      const unsigned ndat = voltages->get_ndat();
      const unsigned offset = voltages->get_input_sample ();

      if (argand)
      {
        assert (ndim==2);
        assert (npol<=2);

        float* data[2];
        
        for (unsigned ipol=0; ipol < npol; ipol++)
          data[ipol] = voltages->get_datptr (0, ipol);

        for (uint64_t idat=0; idat<ndat; idat++)
        {
          for (unsigned ipol=0; ipol < npol; ipol++)
            cout << data[ipol][idat*2] << " " << data[ipol][idat*2+1] << " ";
          cout << endl;
        }
        continue;
      }

      for (unsigned ichan=0; ichan<nchan; ++ichan)
      {
	for (unsigned ipol=0; ipol<npol; ++ipol)
        {
#if 0
          float* data = voltages->get_datptr (ichan, ipol);
          for (uint64_t idat=0; idat<ndim*ndat; idat++)
              cout << ichan << " " << ipol << " "
                   << offset+idat << " " << data[idat] << endl;
#else
          for (unsigned idim=0; idim<ndim; ++idim)
          {
	    float* data = voltages->get_datptr (ichan, ipol) + idim;

	    for (uint64_t idat=0; idat<ndat; idat++)
	      cout << ichan << " " << ipol << " " << idim << " "
                   << offset+idat << " " << data[idat*ndim] << endl;
          }
#endif
	}
      }
    }

    if (verbose)
      cerr << "end of data file " << filenames[ifile] << endl;
  }
  catch (string& error) {
    cerr << error << endl;
  }
  
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

