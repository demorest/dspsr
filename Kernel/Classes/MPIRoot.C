#include <stdio.h>
#include <stdlib.h>
#include <values.h>

#include "dsp/MPIRoot.h"
#include "dsp/BitSeries.h"

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

  mpiPack_size (*loader.get_info(), comm, &temp_size);
  total_size += temp_size;

  MPI_Pack_size (1, MPI_UInt64, comm, &temp_size); 
  total_size += temp_size;  // block_size

  MPI_Pack_size (1, MPI_UNSIGNED, comm, &temp_size); 
  total_size += temp_size;  // resolution

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
  mpiPack (*loader.get_info(), outbuf, outcount, position, comm);

  uint64 block_size;
  block_size = loader.get_block_size ();
  MPI_Pack (&block_size, 1, MPI_UInt64, outbuf, outcount, position, comm);

  unsigned resolution = loader.get_resolution ();
  MPI_Pack (&resolution, 1, MPI_UNSIGNED, outbuf, outcount, position, comm);

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
  dsp::Observation* obs = const_cast<dsp::Observation*> (loader->get_info());

  mpiUnpack (inbuf, insize, position, obs, comm);

  uint64 block_size;
  MPI_Unpack (inbuf, insize, position, &block_size, 1, MPI_UInt64, comm);
  loader->set_block_size (block_size);

  unsigned resolution;
  MPI_Unpack (inbuf, insize, position, &resolution, 1, MPI_UNSIGNED, comm);
  loader->set_resolution (resolution);

  return 0;
}

// ////////////////////////////////////////////////////////////////////////////
//
//
//
dsp::MPIRoot::MPIRoot (MPI_Comm _comm)
 : Input ("MPIRoot")
{
  comm = _comm;

  MPI_Comm_rank (comm, &mpi_rank);
  MPI_Comm_size (comm, &mpi_size);

  mpi_root = -1;
  mpi_tag = 1;

  data_request = MPI_REQUEST_NULL;
  ready_request = MPI_REQUEST_NULL;

  ready = 0;

  async_buf = 0;
  async_buf_size = 0;
  pack_size = 0;
  data_size = 0;

  end_of_data = true;
}


dsp::MPIRoot::~MPIRoot ()
{
  if (async_buf) delete [] async_buf; async_buf = NULL;
}

void dsp::MPIRoot::set_root (int root)
{
  if (root < 0 || root >= mpi_size)
    throw Error (InvalidParam, "dsp::MPIRoot::set_root",
		 "invalid rank=%d.  mpi_size=%d", root, mpi_size);

  mpi_root = root;
}

void dsp::MPIRoot::set_Input (Input* _input)
{
  if (!_input)
    throw Error (InvalidParam, "dsp::MPIRoot::set_Input", "no input");

  input = _input;
  copy (input);
}

void dsp::MPIRoot::set_resolution (unsigned _resolution)
{
  resolution = _resolution;
}

// ////////////////////////////////////////////////////////////////////////////
//
//
//
/*! This method sends the essential information, such as source, start_time
  and block_size, to all of the nodes using mpiBcast */
void dsp::MPIRoot::prepare ()
{
  if (mpi_rank == mpi_root && info.get_nbytes(get_block_size()) > MAXINT)
    throw Error (InvalidState, "dsp::MPIRoot::bcast_setup",
		 "block_size="UI64" will result\n"
		 "\t\tin buffer size greater than MAXINT=%d\n",
		 get_block_size(), MAXINT);

  mpiBcast (this, 1, mpi_root, comm);

  if (verbose && mpi_rank != mpi_root)
    cerr << "dsp::MPIRoot::prepare rank=" << mpi_rank << " block_size="
	 << get_block_size() << endl;

  size_asyncspace ();

  if (data_request != MPI_REQUEST_NULL)
    MPI_Request_free (&data_request);

  data_request = MPI_REQUEST_NULL;

  if (mpi_rank == mpi_root)
    // post the first receive ready request
    MPI_Irecv (&ready, 1, MPI_INT, MPI_ANY_SOURCE, mpi_tag,
	       comm, &ready_request);

  else {
    // post the first receive request
    MPI_Irecv (async_buf, pack_size, MPI_PACKED, mpi_root, mpi_tag,
               comm, &data_request);

    ready = 1;

    // post the first send ready request
    MPI_Isend (&ready, 1, MPI_INT, mpi_root, mpi_tag, comm, &ready_request);
  }

  end_of_data = false;
}


//! End of data
bool dsp::MPIRoot::eod()
{
  return end_of_data;
}

// ////////////////////////////////////////////////////////////////////////////
//
//
//
void dsp::MPIRoot::size_asyncspace ()
{
  int temp_size = 0;
  MPI_Pack_size (1, MPI_Int64, comm, &temp_size);
  pack_size = temp_size;
  MPI_Pack_size (1, MPI_UInt64, comm, &temp_size);
  pack_size += temp_size;
  MPI_Pack_size (1, MPI_UNSIGNED, comm, &temp_size);
  pack_size += temp_size;

  // the total size of nbytes has already been double checked in bcast_setup
  // the extra two bytes enable time sample resolution features
  data_size = info.get_nbytes( get_block_size() ) + 2;

  MPI_Pack_size (data_size, MPI_CHAR, comm, &temp_size);
  pack_size += temp_size;
  
  if (async_buf_size < pack_size) {
    if (async_buf) delete [] async_buf; async_buf = NULL;

    async_buf = new char [pack_size];
    if (async_buf == NULL)
      throw Error (BadAllocation, "dsp::MPIRoot::size_asyncspace",
		   "mpi_rank=%d - error allocate "I64" bytes",
		   mpi_rank, pack_size);

    async_buf_size = pack_size;
  }
}

/*! \param request the asynchronous transfer request to use in MPI_Wait
    \param receive MPI_Status fields are checked only on receive requests
 */
void dsp::MPIRoot::wait (MPI_Request& request, bool receive)
{
  if (request == MPI_REQUEST_NULL)
    return;

  int mpi_err = MPI_Wait (&request, &status);
  request = MPI_REQUEST_NULL;

  check ("dsp::MPIRoot::wait", mpi_err, status, receive);
}

void dsp::MPIRoot::check (const char* method, int mpi_err, MPI_Status& _status,
                          bool receive)
{
  int err_len;
  if (mpi_err != MPI_SUCCESS) {
    MPI_Error_string (mpi_err, mpi_errstr, &err_len);
    throw Error (FailedCall, method, "rank=%d MPI_Wait %s\n",
		 mpi_rank, mpi_errstr);
  }

  if (status.MPI_ERROR != MPI_SUCCESS) {
    MPI_Error_string (status.MPI_ERROR, mpi_errstr, &err_len);
    throw Error (FailedCall, method, "rank=%d MPI_Wait MPI_Status %s\n",
		 mpi_rank, mpi_errstr);
  }

  if (receive && status.MPI_TAG != mpi_tag)
    throw Error (InvalidState, method,
		 "rank=%d MPI_Wait MPI_Status::MPI_TAG=%d != mpi_tag=%d",
		 mpi_rank, status.MPI_TAG, mpi_tag);

}

int dsp::MPIRoot::next_destination ()
{
  ensure_root ("dsp::MPIRoot::next_destination");

  wait (ready_request, true);

  // post the next receive ready request
  MPI_Irecv (&ready, 1, MPI_INT, MPI_ANY_SOURCE, mpi_tag,
	     comm, &ready_request);

  return status.MPI_SOURCE;
}

// ////////////////////////////////////////////////////////////////////////////
//
//
//
void dsp::MPIRoot::send_data (BitSeries* data, int dest)
{
  ensure_root ("dsp::MPIRoot::next_destination");

  if (verbose)
    cerr << "MPIRoot::send_data dest="<< dest <<" BitSeries*="<< data <<endl;

  if (!async_buf)
    throw Error (InvalidState, "MPIRoot::send_data", "not prepared");

  if ((dest == mpi_rank) || (dest < 0) || (dest >= mpi_size))
    throw Error (InvalidParam, "MPIRoot::send_data",
                 "invalid dest=%d. mpi_root=%d mpi_size=%d",
		 dest, mpi_root, mpi_size);

  int nbytes = 0;

  if (data)
    nbytes = data->get_nbytes();

  else if (verbose)
    cerr << "dsp::MPIRoot::send_data sending end of data" << endl;

  if (nbytes > data_size)
    throw Error (InvalidParam, "dsp::MPIRoot::send_data",
		 "invalid nbytes=%d > data_size=%d)",
		 nbytes, data_size);

  // ensure that the asynchronous send buffer is free
  wait (data_request, false);

  int64 start_sample = 0;
  uint64 request_ndat = 0;
  unsigned request_offset = 0;

  char* datptr = 0;

  if (data)  {
    start_sample = data->get_input_sample();
    request_ndat = data->get_request_ndat ();
    request_offset = data->get_request_offset ();

    datptr = (char*) data->get_rawptr();
  }

  int position = 0;
  MPI_Pack (&start_sample, 1, MPI_Int64, async_buf, pack_size,
	    &position, comm);
  MPI_Pack (&request_ndat, 1, MPI_UInt64, async_buf, pack_size,
	    &position, comm);
  MPI_Pack (&request_offset, 1, MPI_UNSIGNED, async_buf, pack_size,
	    &position, comm);

  if (data)
    MPI_Pack (datptr, nbytes, MPI_CHAR, async_buf, pack_size,
	      &position, comm);

  MPI_Isend (async_buf, position, MPI_PACKED, dest, mpi_tag, 
	     comm, &data_request);

  if (data_request == MPI_REQUEST_NULL)
    throw Error (InvalidState, "dsp::MPIRoot::send_data"
		 "Unexpected MPI_REQUEST_NULL from MPI_Isend");

}

// ////////////////////////////////////////////////////////////////////////////
//
//
//
void dsp::MPIRoot::load_data (BitSeries* data)
{
  if (async_buf == NULL)
    throw Error (InvalidState, "dsp::MPIRoot::load_data", 
		 "buffer not prepared.  call bcast_setup first.");

  wait (ready_request, false);
  wait (data_request, true);

  int count;
  MPI_Get_count (&status, MPI_PACKED, &count);

  int header_size = pack_size - data_size;
  int received = count - header_size;

  if (received < 1) {
    end_of_data = true;
    if (verbose)
      cerr << "MPIRoot::load_data end of data" << endl;
  }

  int64 start_sample = 0;
  uint64 request_ndat = 0;
  unsigned request_offset = 0;

  char* datptr = (char*) data->get_rawptr();

  int position = 0;

  MPI_Unpack (async_buf, pack_size, &position, &start_sample, 1, 
	      MPI_Int64, comm);
  MPI_Unpack (async_buf, pack_size, &position, &request_ndat, 1, 
	      MPI_UInt64, comm);
  MPI_Unpack (async_buf, pack_size, &position, &request_offset, 1, 
	      MPI_UNSIGNED, comm);

  MPI_Unpack (async_buf, pack_size, &position, datptr, received,
	      MPI_CHAR, comm);

  // by calling the Input::seek and set_block_size methods, the
  // load_sample and resolution_offset attributes will be properly set
  // and passed on to the output BitSeries
  seek (start_sample + request_offset, SEEK_SET);
  set_block_size (request_ndat);
	       
  // post for the next receive
  MPI_Irecv (async_buf, pack_size, MPI_PACKED, mpi_root, mpi_tag,
             comm, &data_request);

  ready = 1;

  // post the next send ready request
  MPI_Isend (&ready, 1, MPI_INT, mpi_root, mpi_tag, comm, &ready_request);
}


void dsp::MPIRoot::serve (BitSeries* data)
{
  ensure_root ("dsp::MPIRoot::serve");

  if (!input)
    throw Error (InvalidState, "dsp::MPIRoot::serve", "input not set");

  if (!data)
    data = new dsp::BitSeries;

  input->set_output (data);
  

  while (!input->eod ()) {

    if (verbose)
      cerr << "dsp::MPIRoot::serve loading data" << endl;
    input->operate ();

    if (verbose)
      cerr << "dsp::MPIRoot::serve waiting for next ready node" << endl;

    int destination = next_destination ();

    if (verbose)
      cerr << "dsp::MPIRoot::serve sending data to " << destination << endl;

    send_data (data, destination);

  }

  if (verbose)
    cerr << "dsp::MPIRoot::serve end of data" << endl;

  // after end of data, send empty buffer to each node
  for (int irank=0; irank < mpi_size; irank++)
    if (irank != mpi_rank)
      send_data (0, irank);

}

void dsp::MPIRoot::ensure_root (const char* method) const
{
  if (mpi_size <= 1)
    throw Error (InvalidState, method, "mpi_size=%d", mpi_size);

  if (mpi_rank != mpi_root)
    throw Error (InvalidState, method, "mpi_rank=%d != mpi_root=%d",
		 mpi_rank, mpi_root);
}
