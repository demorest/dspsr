/*

 */

#ifndef __dsp_CASPSRUnpacker_h
#define __dsp_CASPSRUnpacker_h

#include "dsp/EightBitUnpacker.h"
#include "ThreadContext.h"

namespace dsp {
  
  class CASPSRUnpacker : public HistUnpacker
  {
  public:

    //! Constructor
    CASPSRUnpacker (const char* name = "CASPSRUnpacker");
    ~CASPSRUnpacker ();

    //! Cloner (calls new)
    virtual CASPSRUnpacker * clone () const;

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const;

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

  protected:
    
    Reference::To<BitTable> table;

    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    void unpack ();

    void unpack (uint64_t ndat, const unsigned char* from, 
		             float* into, const unsigned fskip,
		             unsigned long* hist);

    void * gpu_stream;

    void unpack_on_gpu ();

    unsigned get_resolution ()const ;

  private:

    ThreadContext * context;

    unsigned n_threads;

    unsigned thread_count;

    bool device_prepared;

    bool single_thread;

    void unpack_single_thread ();

    //! cpu_unpacker_thread ids
    std::vector <pthread_t> ids;

    //! Signals the CPU threads to start
    void start_threads ();

    //! Waits for the CPU threads to complete 
    void wait_threads ();

    //! Stops the CPU threads
    void stop_threads ();

    //! Joins the CPU threads
    void join_threads ();

    //! sk_thread calls thread method
    static void* cpu_unpacker_thread (void*);

    //! The CPU CASPSR Unpacker thread
    void thread ();

    enum State { Idle, Active, Quit };

    //! overall state
    State state;

    //! sk_thread states
    std::vector <State> states;

    //! maximum number of GPU threads per block
    int threadsPerBlock;

  };
}

#endif
