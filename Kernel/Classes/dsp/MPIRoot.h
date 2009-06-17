//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __MPIRoot_h
#define __MPIRoot_h

#include <mpi.h>

#define ACTIVATE_MPI 1
#include "dsp/Input.h"


namespace dsp {

  //! Loads BitSeries data using the MPI communications protocol
  class MPIRoot : public Input {
    
  public:
    
    friend class MPIServer;

    //! Construct from MPI_Bcast
    MPIRoot (MPI_Comm comm);

    //! Destructor
    virtual ~MPIRoot ();
    
    //! Get the number of nodes in communicator
    int get_size () const { return mpi_size; }

    //! Get the rank of this process within the communicator
    int get_rank () const { return mpi_rank; }

    //! Get the rank of process sending data
    int get_root () const { return mpi_root; }
    //! Set the rank of process sending data
    void set_root (int root);

    //! Get the tag used for all communication
    int get_tag () const { return mpi_tag; }
    //! Set the tag used for all communication
    void set_tag (int tag) { mpi_tag = tag; }

    //! Setting the block_size requires resizing the buffer
    void set_block_size (uint64_t _size);

    //! Prepare for sending or receiving from root node
    void prepare ();

    //! End of data
    bool eod();

    /** @name mpi_root methods
     *  These methods are used only by the MPIRoot instance for which
     *  mpi_rank == mpi_root
     */
    //@{

    //! Set the source from which input data will be read
    void set_Input (Input* input);

    //! Serve the data from Input
    void serve (BitSeries* bitseries = 0);

    //@}

    //! Provide access to resolution attribute (required in mpiUnpack)
    void set_resolution (unsigned resolution);

  protected:

    //! Method that is p.v. in dsp::Input so that dsp::Seekable can set its 'end_of_data' flag.  I don't know what MPIRoot should do in it, if anything.  HSK 25/10/03
    virtual void set_eod(bool _eod){/* HSK doesn't know what to put here*/}

    //! Send the next block using MPI_Isend
    void send_data (BitSeries* data, int dest);

    //! Load the next block using MPI_Irecv
    void load_data (BitSeries* data);

    //! Wait for the MPI_Request to complete and check the MPI_Status
    void wait (MPI_Request& request, bool receive);

    //! Returns the rank of the next node ready to receive data
    int next_destination ();

    //! resize the pack_buf
    void size_pack_buffer ();

    //! Communicator in which data will be sent and received
    MPI_Comm comm;

    //! Number of nodes in communicator
    int mpi_size;

    //! Rank of this process within the communicator
    int mpi_rank;

    //! Rank of process sending data
    int mpi_root;
 
    //! Tag used for all communication
    int mpi_tag;

    //! Automatically request the next data chunk
    bool auto_request;

    //! Handle to the asynchronous send/recv of data
    MPI_Request data_request;

    //! Handle to the asynchronous send/recv of ready flag
    MPI_Request ready_request;

    //! Status of the last call to wait
    MPI_Status status;

    //! The ready flag
    int ready;

    //! Buffer used to store asynchronous send/recv
    char* pack_buf;

    //! Size of the above buffer
    int pack_buf_size;

    //! Size actually needed
    int pack_size;

    //! MPI_Pack_size of the header in the buffer
    int min_header_size;

    //! Size of the data in bytes
    int data_size;

    //! End of data
    bool end_of_data;

    //! End of data has been sent to this node
    vector<bool> eod_sent;

    //! The source from which input data will be read
    Reference::To<Input> input;

    //! initialize variables
    void init();

    //! request ready-for-data from any node
    void request_ready ();

    //! request data from the root
    void request_data ();

    //! wait for data to be received and return the data count
    int receive_data ();

    //! when mpi_rank == mpi_root, load data from input
    void load_data ();

    //! ensure that this instance is the root and that mpi_size > 1
    void ensure_root (const char* method) const;

    //! ensure that this instance is not the root and that eof = false
    void ensure_receptive (const char* method) const;

    //! throw an Error exception if mpi_err != MPI_SUCCESS
    void check_error (int mpi_err, const char* call, const char* method);

    //! verify that MPI_Wait returns as expected after MPI_Irecv
    void check_status (MPI_Status& mpi_status, const char* method);

    //! verify that the block_size is withing MPI size limits
    void check_block_size (const char* method);

  };

}


//! Return the size required to mpiPack an MPIRoot
int mpiPack_size (const dsp::MPIRoot&, MPI_Comm comm, int* size);

//! Pack an MPIRoot into outbuf
int mpiPack (const dsp::MPIRoot&, void* outbuf, int outcount,
	     int* position, MPI_Comm comm);

//! Unpack an MPIRoot from inbuf
int mpiUnpack (void* inbuf, int insize, int* position, 
	       dsp::MPIRoot*, MPI_Comm comm);


#endif // !defined(__SeekInput_h)
