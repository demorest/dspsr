//-*-C++-*-

#ifndef __MPIRoot_h
#define __MPIRoot_h

#include <mpi.h>

#define ACTIVATE_MPI 1
#include "dsp/Input.h"

namespace dsp {

  //! Loads BitSeries data using the MPI communications protocol
  class MPIRoot : public Input {
    
  public:
    
    //! Construct from MPI_Bcast
    MPIRoot (MPI_Comm comm);

    //! Destructor
    virtual ~MPIRoot ();
    
    //! Prepare for sending or receiving from root node
    void bcast_setup (int root_node);

    //! End of data
    bool eod();
    
    //! Send the next block using MPI_Isend
    void send_data (BitSeries* data, int dest, int nbytes = 0);

    //! Load the next block using MPI_Irecv
    void load_data (BitSeries* data);

    //! MPI load does not support seeking
    void seek (int64 offset, int whence = 0);

    void serve_data (Input* input, BitSeries* data = 0);

  protected:

    //! Wait on the status of the previous MPI_Request, return count
    int wait ();

    //! resize the async_buf
    void size_asyncspace ();

    //! Communicator in which data will be sent and received
    MPI_Comm comm;

    //! Number of nodes in communicator
    int mpi_size;

    //! Node number of this process within the communicator
    int mpi_self;

    //! Node number of process sending data
    int mpi_root;
 
    //! Tag used for all communication
    int mpi_tag;

    //! status of the asynchronous send/recv
    MPI_Request request;

    //! Buffer used to store asynchronous send/recv
    char* async_buf;

    //! Size of the above buffer
    int64 async_buf_size;

    //! Size actually needed
    int64 pack_size;

    //! End of data
    bool end_of_data;

    //! initialize variables
    void init();

  };

}

//! Return the size required to mpiPack an MPIRoot
int mpiPack_size (const dsp::MPIRoot&, MPI_Comm comm, int* size);

//! Pack an MPIRoot into outbuf
int mpiPack (const dsp::MPIRoot&,
	     void* outbuf, int outcount, int* position, MPI_Comm comm);

//! Unpack an MPIRoot from inbuf
int mpiUnpack (void* inbuf, int insize, int* position, 
	       dsp::MPIRoot*, MPI_Comm comm);


#endif // !defined(__SeekInput_h)
