/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <iostream>
#include <unistd.h>

#include "dsp/TestInput.h"
#include "dsp/MPIServer.h"
#include "dsp/MPIRoot.h"
#include "dsp/File.h"

#include "string_utils.h"
#include "dirutil.h"
#include "Error.h"

static char* args = "b:vV";

void usage ()
{
  cout << "test_MPIRoot - test the MPIRoot class\n"
    "Usage: test_MPIRoot [" << args << "] file1 [file2 ...] \n"
    " -b block size  the number of time samples loaded\n"
       << endl;
}

int main (int argc, char** argv) 

{ try {

  MPI_Init (&argc, &argv);
   // MPI_Errhandler_set (MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  int  len_name;
  char mpi_name [MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name (mpi_name, &len_name);

  cerr << "test_MPIRoot running on " << mpi_name << endl;

  // size of the data blocks to be read
  unsigned block_size = 4096;

  // rank of root node
  int mpi_root = 0;

  // verbosity flag
  bool verbose = false;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'V':
      dsp::Observation::verbose = true;
      dsp::Operation::verbose = true;

    case 'v':
      verbose = true;
      break;

    case 'b':
      block_size = atoi (optarg);
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  vector <string> filenames;
  for (int ai=optind; ai<argc; ai++)
    dirglob (&filenames, argv[ai]);

  if (filenames.size() == 0) {
    usage ();
    return -1;
  }

  if (verbose)
    cerr << "Creating TestInput instance" << endl;
  dsp::TestInput test;
  test.set_block_size (block_size);

  if (verbose)
    cerr << "Creating MPIRoot instances" << endl;

  Reference::To<dsp::MPIRoot> mpi_a = new dsp::MPIRoot (MPI_COMM_WORLD);
  Reference::To<dsp::MPIRoot> mpi_b = new dsp::MPIRoot (MPI_COMM_WORLD);

  mpi_a->set_root (mpi_root);
  mpi_a->set_tag (2);

  mpi_b->set_root (mpi_root);
  mpi_a->set_tag (3);

  int retval = 0;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) {

    if (mpi_a->get_root() == mpi_a->get_rank()) {

      if (verbose)
	cerr << "opening data file " << filenames[ifile] << endl;

      Reference::To<dsp::Input> input_a = dsp::File::create (filenames[ifile],0);
      Reference::To<dsp::Input> input_b = dsp::File::create (filenames[ifile],0);

      mpi_a -> set_Input (input_a);
      mpi_a -> prepare ();

      mpi_b -> set_Input (input_b);
      mpi_b -> prepare ();

      dsp::MPIServer server;

      server.manage (mpi_a);
      server.manage (mpi_b);

      server.serve ();

      cerr << "end of data file " << filenames[ifile] << endl << endl;

    }

    else {

      mpi_a -> prepare ();
      mpi_b -> prepare ();

      // note that TestInput::runtest knows only the Input base class
      test.runtest (mpi_a, mpi_b);

      if (test.get_errors() == 0)
        cerr << "test_MPIRoot successful completion.  no errors." << endl;
      else {
        cerr << "test_MPIRoot test failed" << endl;
        retval = -1;
      }
    }

  }

  MPI_Finalize ();
  return retval;
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

}
