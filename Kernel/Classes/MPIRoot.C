#include <stdio.h>
#include <stdlib.h>
#include <values.h>

#include "dsp/MPIRoot.h"
#include "dsp/Timeseries.h"

#include "stdmpi.h"
#include "portable_mpi.h"
#include "genutil.h"

// #define _DEBUG 1
static char mpi_errstr [MPI_MAX_ERROR_STRING];

// ////////////////////////////////////////////////////////////////////////////
//
//
//
//! Return the size required to mpiPack an MPIRoot
int mpiPack_size (const dsp::MPIRoot& loader, MPI_Comm comm, int* size)
{
  int total_size = 0;
  int temp_size = 0;

  mpiPack_size ((const dsp::Observation&) loader, comm, &temp_size);
  total_size += temp_size;

  MPI_Pack_size (1, MPI_UInt64, comm, &temp_size); 
  total_size += temp_size;  // block_size

  *size = total_size;
  return 1; // no error, dynamic
}

// ////////////////////////////////////////////////////////////////////////////
//
//
//
//! Pack an MPIRoot into outbuf
int mpiPack (const dsp::MPIRoot& loader,
	     void* outbuf, int outcount, int* position, MPI_Comm comm)
{
  mpiPack ((const dsp::Observation&) loader, outbuf, outcount, position, comm);

  uint64 block_size;
  block_size = loader.get_block_size ();
  MPI_Pack (&block_size, 1, MPI_UInt64, outbuf, outcount, position, comm);

  return 0;
}

// ////////////////////////////////////////////////////////////////////////////
//
//
//
//! Unpack an MPIRoot from inbuf
int mpiUnpack (void* inbuf, int insize, int* position, 
	       dsp::MPIRoot* loader, MPI_Comm comm)
{
  mpiUnpack (inbuf, insize, position, (dsp::Observation*) loader, comm);

  uint64 block_size;
  MPI_Unpack (inbuf, insize, position, &block_size, 1, MPI_UInt64, comm);
  loader->set_block_size (block_size);

  return 0;
}

// ////////////////////////////////////////////////////////////////////////////
//
//
//
dsp::MPIRoot::MPIRoot (MPI_Comm _comm)
{
  comm = _comm;

  MPI_Comm_rank (comm, &mpi_self);
  MPI_Comm_size (comm, &mpi_size);

  mpi_root = -1;
  mpi_tag = 69;

  request = MPI_REQUEST_NULL;

  async_buf = 0;
  async_buf_size = 0;
  pack_size = 0;

  end_of_data = false;
}

// ////////////////////////////////////////////////////////////////////////////
//
//
//
/*! This method sends the essential information, such as source, start_time
  and block_size, to all of the nodes using mpiBcast */
void dsp::MPIRoot::bcast_setup (int root_node)
{
  mpi_root = root_node;

  if (mpi_self == mpi_root && info.nbytes(block_size) > MAXINT)
    throw_str ("MPIRoot::bcast_setup block_size="UI64" will result\n"
	       "\t\tin buffer size greater than MAXINT:%d\n",
	       block_size, MAXINT);

  mpiBcast (this, 1, mpi_root, comm);

#ifdef _DEBUG
  fprintf(stderr, "MPIRoot::%d:bcast_setup start time: %s\n",
	  mpi_self, source_start_time.printall());
  fprintf(stderr, "MPIRoot::%d:bcast_setup block size: %lu\n",
	  mpi_self, block_size);
  fprintf(stderr, "MPIRoot::%d:bcast_setup sampl rate: %lf\n",
	  mpi_self, rate);
#endif

  size_asyncspace ();

  if (request != MPI_REQUEST_NULL)
    MPI_Request_free (&request);

  request = MPI_REQUEST_NULL;

  if (mpi_self != mpi_root)
    // post the first receive request
    MPI_Irecv (async_buf, pack_size, MPI_PACKED, mpi_root, mpi_tag,
               comm, &request);
}

// ////////////////////////////////////////////////////////////////////////////
//
//
//
void dsp::MPIRoot::size_asyncspace ()
{
  int temp_size = 0;
  MPI_Pack_size (1, MPI_UNSIGNED_LONG, comm, &temp_size);
  pack_size = temp_size;

  // the total size of nbytes has already been double checked in bcast_setup
  MPI_Pack_size ((int)info.nbytes(block_size), MPI_CHAR, comm, &temp_size);
  pack_size += temp_size;
  
  if (async_buf_size < pack_size) {
    if (async_buf) delete [] async_buf; async_buf = NULL;

    async_buf = new char [pack_size];
    if (async_buf == NULL) {
      throw_str ("MPIRoot::%d:size_asyncspace - error allocate "I64" bytes",
		 mpi_self, pack_size);
    }
    async_buf_size = pack_size;
  }
}

// ////////////////////////////////////////////////////////////////////////////
//
//
//
int dsp::MPIRoot::wait ()
{
  if (request == MPI_REQUEST_NULL)
    return 0;

  MPI_Status status;

  int mpi_err = MPI_Wait (&request, &status);
  request = MPI_REQUEST_NULL;

  int err_len;
  if (mpi_err != MPI_SUCCESS) {
    MPI_Error_string (mpi_err, mpi_errstr, &err_len);
    throw_str ("MPIRoot::%d:wait - MPI_Wait %s\n",
	       mpi_self, mpi_errstr);
  }

  if (status.MPI_ERROR != MPI_SUCCESS) {
    MPI_Error_string (status.MPI_ERROR, mpi_errstr, &err_len);
    throw_str ("MPIRoot::%d:wait - status.MPI_ERROR %s\n",
	       mpi_self, mpi_errstr);
  }

  return status.count;
}

// ////////////////////////////////////////////////////////////////////////////
//
//
//
void dsp::MPIRoot::send_data (Timeseries* data, int dest, int nbytes)
{
  if (verbose)
    cerr << "MPIRoot::send_data dest="<< dest <<" nbytes="<< nbytes <<endl;

  if (!async_buf)
    throw_str ("MPIRoot::send_data not prepared");

  if ((dest == mpi_self) || (dest < 0) || (dest >= mpi_size))
    throw_str ("MPIRoot::send_data invalid dest");

  if (nbytes < 0 || nbytes > (int) data->nbytes() || nbytes > pack_size)
    throw_str ("MPIRoot::send_data invalid nbytes=%d"
	       " (input.nbytes=%d pack_size=%d)",
	       nbytes, (int) data->nbytes(), pack_size);

  // ensure that the asynchronous send/recv buffer is free
  wait ();

  unsigned long start_sample = data->get_input_sample();
  char* datptr = (char*) data->get_rawptr();

  int position = 0;
  MPI_Pack (&start_sample, 1, MPI_UNSIGNED_LONG, async_buf, pack_size,
	    &position, comm);
  MPI_Pack (datptr, nbytes, MPI_CHAR, async_buf, pack_size,
	    &position, comm);

  MPI_Isend (async_buf, position, MPI_PACKED, dest, mpi_tag, 
	     comm, &request);

  if (request == MPI_REQUEST_NULL)
    throw_str ("MPIRoot::send_data"
	       " Unexpected MPI_REQUEST_NULL from MPI_Isend");

}

// ////////////////////////////////////////////////////////////////////////////
//
//
//
void dsp::MPIRoot::load_data (Timeseries* data)
{
  if (async_buf == NULL)
    throw_str ("MPIRoot::load_data not prepared");

  if (request == MPI_REQUEST_NULL)
    // no receives have yet been posted!  get on it
    MPI_Irecv (async_buf, pack_size, MPI_PACKED, mpi_root, mpi_tag,
	       comm, &request);

  int count = wait ();

  int received = count - sizeof(unsigned long);

  if (received < 1) {
    end_of_data = true;
    if (verbose)
      cerr << "MPIRoot::load_data end of data" << endl;
  }

  unsigned long start_sample = 0;
  char* datptr = (char*) data->get_rawptr();

  int position = 0;
  MPI_Unpack (async_buf, pack_size, &position, &start_sample, 1, 
	      MPI_UNSIGNED_LONG, comm);
  MPI_Unpack (async_buf, pack_size, &position, datptr, received,
	      MPI_CHAR, comm);
  
  data->change_start_time (start_sample);
  data->set_ndat (info.nsamples(received));

  // post for the next receive
  MPI_Irecv (async_buf, pack_size, MPI_PACKED, mpi_root, mpi_tag,
             comm, &request);

}
