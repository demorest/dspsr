#include <stdio.h>
#include <stdlib.h>
#include <values.h>

#include "dsp/MPITrans.h"
#include "dsp/MPIRoot.h"
#include "dsp/BitSeries.h"

#include "stdmpi.h"
#include "portable_mpi.h"
#include "genutil.h"


dsp::MPITrans::MPITrans (MPI_Comm _comm) 
  : MPIRoot (_comm)
{
  
  auto_request = false;

}

void dsp::MPITrans::send_data (BitSeries* data, int dest)
{
  
  MPIRoot::send_data(data, dest);
  
}

void dsp::MPITrans::load_data (BitSeries* data)
{

  MPIRoot::load_data(data);

}

void dsp::MPITrans::load_data ()
{

  MPIRoot::load_data();

}

int dsp::MPITrans::next_destination ()
{
  
  return MPIRoot::next_destination();
  
}

void dsp::MPITrans::request_data ()
{
  
  MPIRoot::request_data ();
  
}

void dsp::MPITrans::request_ready (int node)
{
  // ensure_root ("dsp::MPIRoot::request_ready");

  if (ready_request != MPI_REQUEST_NULL)
    throw Error (InvalidState, "dsp::MPIRoot::request_ready",
                 "ready_request already pending");

  // post the receive ready-for-data request
  MPI_Irecv (&ready, 1, MPI_INT, node, mpi_tag, comm, &ready_request);
}
