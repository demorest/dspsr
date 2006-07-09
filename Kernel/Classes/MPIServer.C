/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/MPIServer.h"
#include "dsp/MPIRoot.h"
#include "dsp/BitSeries.h"

//! Default constructor
dsp::MPIServer::MPIServer ()
{
}

//! Destructor
dsp::MPIServer::~MPIServer ()
{
}

//! Manage the MPIRoot instance
void dsp::MPIServer::manage (MPIRoot* _root)
{
  root.push_back (_root);
}

//! Serve the data from the managed MPIRoot instances
void dsp::MPIServer::serve ()
{
  if (root.size() == 0)
    throw Error (InvalidState, "dsp::MPIServer::serve",
		 "no managed MPIRoot instances");

  unsigned ir = 0;
  unsigned have_data = 0;

  for (ir=0; ir < root.size(); ir++) {

    // ensure that each managed MPIRoot instance is the root node
    root[ir]->ensure_root ("dsp::MPIServer::serve");

    // ensure that each has an Input to serve
    if (!root[ir]->input)
      throw Error (InvalidState, "dsp::MPIServer::serve", "input not set");

    // count the number of inputs with data
    if (!root[ir]->eod())
      have_data ++;
  }

  if (!have_data)
    throw Error (InvalidState, "dsp::MPIServer::serve",
		 "no input data to serve (all eod)");

  Reference::To<BitSeries> data = new dsp::BitSeries;
  vector< MPI_Request > ready( root.size() );

  for (ir=0; ir < root.size(); ir++) {
    // allocate buffer to each MPIRoot input data stream
    root[ir]->input->set_output (data);
  }

  MPI_Status status;

  MPI_Request *requests = &(ready[0]);

  char* method = "dsp::MPIServer::serve";

  while (have_data) {

    have_data = 0;
    for (ir=0; ir < root.size(); ir++) {

      // reference the ready_request attributes of each MPIRoot
      ready[ir] = root[ir]->ready_request;

      // count the number of inputs with data
      if (!root[ir]->eod())
        have_data ++;

    }

    if (!have_data) {
      if (MPIRoot::verbose)
        cerr << "dsp::MPIServer::serve end of data" << endl;
      break;
    }

    int index = 0;
    int mpi_err = MPI_Waitany( root.size(), requests, &index, &status );
    
    if (index == MPI_UNDEFINED)
      throw Error (InvalidState, "dsp::MPIServer::server",
		   "MPI_Waitany index=MPI_UNDEFINED");

    // ensure that all is good
    root[index]->check_error (mpi_err, "MPI_Waitany", method);
    root[index]->check_error (status.MPI_ERROR, "MPI_Waitany status", method);
    root[index]->check_status (status, method);

    // reset the ready-to-receive request handle
    root[index]->ready_request = MPI_REQUEST_NULL;

    int dest = status.MPI_SOURCE;

    if (MPIRoot::verbose)
      cerr << "dsp::MPIServer::serve index=" << index 
	   << " dest=" << dest << endl;

    if ( root[index]->input->eod() ) {

      if (MPIRoot::verbose)
	cerr << "dsp::MPIServer::serve end of data[" << index << "]" << endl;

      root[index]->send_data (0, dest);

    }
    else {
      if (MPIRoot::verbose)
	cerr << "dsp::MPIServer::serve loading data[" << index << "]" << endl;

      root[index]->load_data ();

      if (MPIRoot::verbose)
	cerr << "dsp::MPIServer::serve sending data to " << dest << endl;
      
      root[index]->send_data (data, dest);

    }
  }
}

