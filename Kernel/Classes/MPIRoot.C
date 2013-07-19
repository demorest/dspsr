/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
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

  uint64_t block_size;
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

  uint64_t block_size;
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

  pack_buf = 0;
  pack_buf_size = 0;
  pack_size = 0;
  min_header_size = 0;
  data_size = 0;

  auto_request = true;

  // compute the MPI_Pack_size of the header information
  int temp_size = 0;
  MPI_Pack_size (1, MPI_Int64, comm, &temp_size);
  min_header_size += temp_size;
  MPI_Pack_size (1, MPI_UInt64, comm, &temp_size);
  min_header_size += temp_size;
  MPI_Pack_size (1, MPI_UNSIGNED, comm, &temp_size);
  min_header_size += temp_size;

  end_of_data = true;
}


dsp::MPIRoot::~MPIRoot ()
{
  if (pack_buf) delete [] pack_buf; pack_buf = NULL;
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

void dsp::MPIRoot::set_block_size (uint64_t _size)
{
  bool resize_required = (get_block_size() != _size);

  Input::set_block_size (_size);

  if (!end_of_data && resize_required)  {
    size_pack_buffer ();
    if (ready && mpi_rank != mpi_root && auto_request){
      if(verbose)
        cerr << "dsp::MPIRoot::set_block_size REQUESTING DATA" << endl;
      request_data();
    }
  }
}

// ////////////////////////////////////////////////////////////////////////////
//
//
//
/*! This method sends the essential information, such as source, start_time
  and block_size, to all of the nodes using mpiBcast */
void dsp::MPIRoot::prepare ()
{
  if (!end_of_data)
    throw Error (InvalidState, "dsp::MPIRoot::prepare",
                 "cannot prepare when end_of_data != false");

  if (mpi_rank == mpi_root)
    check_block_size ("dsp::MPIRoot::prepare");

  mpiBcast (this, 1, mpi_root, comm);

  if (verbose && mpi_rank != mpi_root)
    cerr << "dsp::MPIRoot::prepare rank=" << mpi_rank << " block_size="
	 << get_block_size() << endl;

  ready = 0;

  if (get_block_size ())
    size_pack_buffer ();

  end_of_data = false;

  if (mpi_rank == mpi_root) {

    eod_sent.resize (mpi_size);

    for (int inode=0; inode<mpi_size; inode++)
      eod_sent[inode] = false;

    // don't need to send to self
    eod_sent[mpi_rank] = true;

    if( auto_request )
      request_ready ();

  }

  else if (ready && auto_request){
    if(verbose)
      cerr << "dsp::MPIRoot::prepare REQUESTING DATA" << endl;
    request_data ();
  }
}


//! End of data
bool dsp::MPIRoot::eod()
{

  if (!end_of_data && mpi_rank != mpi_root)
    receive_data ();

  return end_of_data;
}

// ////////////////////////////////////////////////////////////////////////////
//
//
//
void dsp::MPIRoot::size_pack_buffer ()
{
  if (data_request != MPI_REQUEST_NULL || 
     (ready_request != MPI_REQUEST_NULL && mpi_rank != mpi_root))
    throw Error (InvalidState, "dsp::MPIRoot::size_pack_buffer",
                 "cannot resize send/recv buffer when message is pending");

  check_block_size ("dsp::MPIRoot::set_block_size");

  pack_size = min_header_size;

  // the extra bytes enable time sample resolution features
  data_size = get_info()->get_nbytes( get_block_size() ) + resolution;

  int temp_size = 0;
  MPI_Pack_size (data_size, MPI_CHAR, comm, &temp_size);
  pack_size += temp_size;
  
  if (pack_buf_size < pack_size) {
    if (pack_buf) delete [] pack_buf; pack_buf = NULL;

    pack_buf = new char [pack_size];
    if (pack_buf == NULL)
      throw Error (BadAllocation, "dsp::MPIRoot::size_pack_buffer",
		   "mpi_rank=%d - error allocate "I64" bytes",
		   mpi_rank, pack_size);

    pack_buf_size = pack_size;
  }

  if (mpi_rank != mpi_root)
    ready = get_block_size();
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

  char* method = "dsp::MPIRoot::wait";

  check_error (mpi_err, "MPI_Wait", method);
  check_error (status.MPI_ERROR, "MPI_Wait MPI_Status", method);

  if (receive)
    check_status (status, method); 
}

void dsp::MPIRoot::check_error (int err, const char* call, const char* method)
{
  int err_len;
  if (err != MPI_SUCCESS) { 
    MPI_Error_string (err, mpi_errstr, &err_len);
    throw Error (FailedCall, method, "rank=%d %s %s\n",
                 mpi_rank, call, mpi_errstr);
  }
}

void dsp::MPIRoot::check_status (MPI_Status& mpi_status, const char* method)
{
  if (mpi_status.MPI_TAG != mpi_tag)
    throw Error (InvalidState, method,
		 "rank=%d MPI_Status::MPI_TAG=%d != mpi_tag=%d",
		 mpi_rank, mpi_status.MPI_TAG, mpi_tag);
}

void dsp::MPIRoot::check_block_size (const char* method)
{
  uint64_t bytes = get_info()->get_nbytes(get_block_size()) + 2 + min_header_size;

  if (bytes > MAXINT)
    throw Error (InvalidState, method, "block_size="UI64" -> buffer_size="UI64
                 " > MAXINT=%d\n", get_block_size(), bytes, MAXINT);
}

int dsp::MPIRoot::next_destination ()
{
  ensure_root ("dsp::MPIRoot::next_destination");

  wait (ready_request, true);

  return status.MPI_SOURCE;
}

void dsp::MPIRoot::request_ready ()
{
  ensure_root ("dsp::MPIRoot::request_ready");

  if (ready_request != MPI_REQUEST_NULL)
    throw Error (InvalidState, "dsp::MPIRoot::request_ready",
                 "ready_request already pending");

  // post the receive ready-for-data request
  MPI_Irecv (&ready, 1, MPI_INT, MPI_ANY_SOURCE, mpi_tag, comm, &ready_request);
}


// ////////////////////////////////////////////////////////////////////////////
//
//
//

void dsp::MPIRoot::load_data ()
{
  ensure_root ("dsp::MPIRoot::load_data");

  if (!input)
    throw Error (InvalidState, "dsp::MPIRoot::load_data", "no input");

  if (!ready)
    return;

  set_block_size (ready);
  input->set_block_size (ready);
  input->operate ();
}


void dsp::MPIRoot::send_data (BitSeries* data, int dest)
{
  ensure_root ("dsp::MPIRoot::send_data");

  if (verbose)
    cerr << "MPIRoot::send_data dest="<< dest <<" BitSeries*="<< data <<endl;

  if (end_of_data)
    throw Error (InvalidState, "MPIRoot::send_data", "end of data");

  if (!pack_buf)
    size_pack_buffer ();

  if ((dest == mpi_rank) || (dest < 0) || (dest >= mpi_size))
    throw Error (InvalidParam, "MPIRoot::send_data",
                 "invalid dest=%d. mpi_root=%d mpi_size=%d",
		 dest, mpi_root, mpi_size);

  // ensure that the asynchronous send buffer is free
  wait (data_request, false);

  int nbytes = 0;
  int64_t start_sample = 0;
  uint64_t request_ndat = 0;
  unsigned request_offset = 0;

  char* datptr = 0;

  if (data)  {
    nbytes = data->get_nbytes();
    start_sample = data->get_input_sample();
    request_ndat = data->get_request_ndat ();
    request_offset = data->get_request_offset ();

    datptr = (char*) data->get_rawptr();
  }
  else {
    if (verbose)
      cerr << "dsp::MPIRoot::send_data sending end of data" << endl;
    eod_sent[dest] = true;

    end_of_data = true;
    for (int inode=0; inode<mpi_size; inode++)
      if (!eod_sent[inode])
        end_of_data = false;
  }

  if (!end_of_data && auto_request)
    request_ready();

  if (nbytes > data_size)
    throw Error (InvalidParam, "dsp::MPIRoot::send_data",
                 "invalid nbytes=%d > data_size=%d)",
                 nbytes, data_size);

  int position = 0;
  MPI_Pack (&start_sample, 1, MPI_Int64, pack_buf, pack_size,
	    &position, comm);
  MPI_Pack (&request_ndat, 1, MPI_UInt64, pack_buf, pack_size,
	    &position, comm);
  MPI_Pack (&request_offset, 1, MPI_UNSIGNED, pack_buf, pack_size,
	    &position, comm);

  if (data)
    MPI_Pack (datptr, nbytes, MPI_CHAR, pack_buf, pack_size,
	      &position, comm);

  MPI_Isend (pack_buf, position, MPI_PACKED, dest, mpi_tag, 
	     comm, &data_request);

  if (data_request == MPI_REQUEST_NULL)
    throw Error (InvalidState, "dsp::MPIRoot::send_data"
		 "Unexpected MPI_REQUEST_NULL from MPI_Isend");

}

// ////////////////////////////////////////////////////////////////////////////
//
//
//

void dsp::MPIRoot::request_data ()
{
  ensure_receptive ("dsp::MPIRoot::request_data");

  if (data_request != MPI_REQUEST_NULL)
    throw Error (InvalidState, "dsp::MPIRoot::request_data",
                 "data_request already pending");

  // post receive data request
  MPI_Irecv (pack_buf, pack_size, MPI_PACKED, mpi_root, mpi_tag, comm, 
             &data_request);

  if (data_request == MPI_REQUEST_NULL)
    throw Error (InvalidState, "dsp::MPIRoot::request_data",
                 "Unexpected MPI_REQUEST_NULL from MPI_Irecv");

  if (ready_request != MPI_REQUEST_NULL)
    throw Error (InvalidState, "dsp::MPIRoot::request_data",
                 "ready_request already pending");

  // post send ready-to-receive request
  MPI_Isend (&ready, 1, MPI_INT, mpi_root, mpi_tag, comm, &ready_request);
  if (ready_request == MPI_REQUEST_NULL)
    throw Error (InvalidState, "dsp::MPIRoot::request_data",
                 "Unexpected MPI_REQUEST_NULL from MPI_Isend");
}


int dsp::MPIRoot::receive_data ()
{
  ensure_receptive ("dsp::MPIRoot::receive_data");

  if (verbose)
    cerr << "dsp::MPIRoot::receive_data" << endl;

  if (data_request)  {
    wait (ready_request, false);
    wait (data_request, true);
  }

  int count;
  MPI_Get_count (&status, MPI_PACKED, &count);

  int64_t start_sample = 0;
  uint64_t request_ndat = 0;
  unsigned request_offset = 0;

  int position = 0;

  MPI_Unpack (pack_buf, pack_size, &position, &start_sample, 1,
              MPI_Int64, comm);
  MPI_Unpack (pack_buf, pack_size, &position, &request_ndat, 1,
              MPI_UInt64, comm);
  MPI_Unpack (pack_buf, pack_size, &position, &request_offset, 1,
              MPI_UNSIGNED, comm);

  int received = count - position;

  if (request_ndat == 0) {
    end_of_data = true;
    if (verbose)
      cerr << "dsp::MPIRoot::receive_data end of data" << endl;
  }

  return received;
}


void dsp::MPIRoot::load_data (BitSeries* data)
{
  ensure_receptive ("dsp::MPIRoot::load_data");

  if (pack_buf == NULL)
    throw Error (InvalidState, "dsp::MPIRoot::load_data", 
		 "pack buffer not ready");

  int received = receive_data ();

  int64_t start_sample = 0;
  uint64_t request_ndat = 0;
  unsigned request_offset = 0;

  char* datptr = (char*) data->get_rawptr();

  int position = 0;

  MPI_Unpack (pack_buf, pack_size, &position, &start_sample, 1, 
	      MPI_Int64, comm);
  MPI_Unpack (pack_buf, pack_size, &position, &request_ndat, 1, 
	      MPI_UInt64, comm);
  MPI_Unpack (pack_buf, pack_size, &position, &request_offset, 1, 
	      MPI_UNSIGNED, comm);

  MPI_Unpack (pack_buf, pack_size, &position, datptr, received,
	      MPI_CHAR, comm);

  // by calling the Input::seek and set_block_size methods, the
  // load_sample and resolution_offset attributes will be properly set
  // and passed on to the output BitSeries
  Input::seek (start_sample + request_offset, SEEK_SET);
  Input::set_block_size (request_ndat);

  data->set_ndat( data->get_nsamples(received) );

  // extra sanity check
  if (data->get_nbytes() != unsigned(received))
    throw Error (InvalidState, "dsp::MPIRoot::load_data", 
		 "BitSeries::nbytes=%d != received=%d",
		 data->get_nbytes(), received);

  data->set_start_time( get_info()->get_start_time() );
  data->change_start_time( start_sample );

  // request the next block of data
  if(auto_request){
    if (verbose)
      cerr << "dsp::MPIRoot::load_data REQUESTING DATA" << endl;
    request_data ();
  }
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

    int dest = next_destination ();

    if (verbose)
      cerr << "dsp::MPIRoot::serve sending data to " << dest << endl;

    send_data (data, dest);

  }

  if (verbose)
    cerr << "dsp::MPIRoot::serve end of data" << endl;

  while (!eod ()) {

    int dest = next_destination ();

    if (verbose)
      cerr << "dsp::MPIRoot::serve sending end of data to " << dest << endl;

    send_data (0, dest);

  }

}

void dsp::MPIRoot::ensure_root (const char* method) const
{
  if (mpi_size <= 1)
    throw Error (InvalidState, method, "mpi_size=%d", mpi_size);

  if (mpi_rank != mpi_root)
    throw Error (InvalidState, method, "mpi_rank=%d != mpi_root=%d",
		 mpi_rank, mpi_root);
}

void dsp::MPIRoot::ensure_receptive (const char* method) const
{
  if (mpi_rank == mpi_root)
    throw Error (InvalidState, method, "mpi_rank=%d == mpi_root=%d",
                 mpi_rank, mpi_root);
  if (end_of_data)
    throw Error (InvalidState, method, "end of data");
}

