/***************************************************************************
 *
 *   Copyright (C) 2009 by Ravi Kumar
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <iostream>
#include <unistd.h>
#include <Jones.h>

#include "dsp/File.h"
#include "dsp/Response.h"

#include "Pulsar/BasicArchive.h"
#include "Pulsar/Database.h"
#include "Pulsar/CalibratorTypes.h"
#include "Pulsar/PolnCalibrator.h"
#include "Pulsar/Backend.h"
#include "Pulsar/Receiver.h"
#include "dsp/PolnCalibration.h"
#include "dsp/Dedispersion.h"

#include "Error.h"

using namespace std;
using namespace Pulsar;

static char* args = "d:vV";

void usage ()
{
  cout << "test_PolnCalibration -d <database_file> <baseband_data_filename>\n"
       << endl;
}




int main (int argc, char** argv) try 
{
  bool verbose = false;

  char* database_filename = 0;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'V':
      dsp::Observation::verbose = true;
      dsp::Operation::verbose = true;
      MJD::verbose = true;

    case 'v':
      verbose = true;
      break;

    case 'd':
      database_filename = optarg;
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  vector <string> filenames;

  for (int ai=optind; ai<argc; ai++)
    filenames.push_back (argv[ai]);

  if (filenames.size() == 0)
  {
    usage ();
    return 0;
  }

  Reference::To<dsp::Input> file;

  Reference::To<Pulsar::PolnCalibrator > pcal;

  Reference::To<Pulsar::Calibrator::Type> type;
  type = new Pulsar::CalibratorTypes::SingleAxis;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try
  {

    unsigned errors = 0;

    if (verbose)
      cerr << "opening data file " << filenames[ifile] << endl;

    file = dsp::File::create (filenames[ifile]);

    dsp::Observation* obs = file->get_info();

    dsp::PolnCalibration* response = new dsp::PolnCalibration;
 
    unsigned nchan = obs->get_nchan();

    response-> set_database_filename (database_filename);
    response-> match (obs, nchan);

    dsp::Dedispersion::verbose = true;

    dsp::Dedispersion* dedisp = new dsp::Dedispersion;
 
    cerr << "try Dedispersion::match" << endl;
    obs->set_dispersion_measure (10.0);
    dedisp->match (obs, nchan);
    cerr << "finish Dedispersion::match" << endl;

   

  }
  catch (string& error) {
    cerr << error << endl;
  }

  return 0;
}

catch (Error& error) {
  cerr << "Error thrown: " << error << endl;
  return -1;
}

catch (string& error) {
  cerr << "exception thrown: " << error << endl;
  return -1;
}

catch (...) {
  cerr << "unknown exception thrown." << endl;
  return -1;
}
