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

