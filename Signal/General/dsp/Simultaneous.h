//-*-C++-*-

#ifndef __baseband_dsp_Simultaneous_h_
#define __baseband_dsp_Simultaneous_h_

#include <vector>
#include <queue>

#include "psr_cpp.h"
#include "Reference.h"

#include "dsp/BasicBuffer.h"
#include "dsp/Operation.h"
#include "dsp/Buffer.h"

namespace dsp {

  //! An extension of dsp::RingBuffer to allow any class to run in a separate thread
  //! Be wary of use in save_data context + CombineThread context
  //! Be wary of changing input/output of the Operation mid-run
  //! HSK 17 January 2005
  class Simultaneous : public Operation {

  public:

    //! Null constructor
    Simultaneous();

    //! Virtual destructor
    virtual ~Simultaneous();

    //! Retrieve a pointer to the Operation
    Operation* get_op(){ return op; }

    //! Inquire whether an Operation is being stored
    bool has_op(){ return op; }

    //! Set the Operation
    void set_op(Operation* _op){ op = _op; }
    
    //! Convenience method to return the Operation
    template<class T>
    T* get();

  protected:

    //! Does the work
    virtual void operation ();

    //! Puts the new input data into the input ringbuffer and puts the
    //! output data processed in the extra thread into the output
    //! so that it's ready to be processed by the next Operation
    //! (If necessary)
    bool new_data();

    //! Copies the new input data into the input ringbuffer if necessary
    bool copy_input_into_ringbuffer();

    //! Puts the output data processed in the extra thread into the output
    //! so that it's ready to be processed by the next Operation
    //! (If necessary)
    bool copy_ringbuffer_into_output();

    //! Returns a free buffer from the input ringbuffer
    Reference::To<dsp::BasicBuffer> get_input_buffer();

    //! Returns a full buffer from the output ringbuffer
    Reference::To<BasicBuffer> get_output_buffer();

    //! Worker function for get_input_buffer() and get_output_buffer()
    Reference::To<BasicBuffer>
    get_buffer(BasicBuffer::BufferStatus status,
	       vector<Reference::To<BasicBuffer> >& buffers);

    //! Calls wait_on_mutex() for the appropriate mutex and cond
    void wait_for_buffer(BasicBuffer::BufferStatus status);

    //! Spawns the extra thread if using threads
    void start_thread();

    //! Destroys the extra thread
    void destroy_thread();

    //! This function runs in the extra thread
    static void* op_loop(Simultaneous* thiz);
 
    //! Makes the thread wait for the call to operate() before running
    void wait_to_synch();
    
    //! Returns whether or not the user has said to stop
    bool can_run();
    
    //! Broadcasts the fact that a buffer status has changed to all threads
    void broadcast_status(BasicBuffer::BufferStatus status);
    
    //! Checks stuff before operate()
    void checks();
    
    //! Calls op->operate() after setting up the input/output buffers
    void run_op();
    
    //! Returns true if the Operation needs an input to be set
    bool op_uses_an_input();
    
    //! Returns true if the Operation needs an output to be set
    bool op_uses_an_output();
    
    //! Makes sure the buffers point to something sane
    void setup_buffers();
    
    //! Worker function that actually calls 'new' for each element in the vector
    template<class T>
    bool newify_buffers(vector<Reference::To<BasicBuffer> >& buffers);

    //! Tries to work out what sort of buffers the input_buffers should be 
    bool newify_input_buffers();

    //! Tries to work out what sort of buffers the input_buffers should be 
    bool newify_output_buffers();

    //! Worker function for newify_input_buffers() and newify_output_buffers()
    bool newify_those_buffers(string typestring,
			      vector<Reference::To<BasicBuffer> >& buffers);

    //! Gives the Operation an output if it needs one
    Reference::To<BasicBuffer> setup_op_output();
    //! Gives the Operation an input if it needs one
    Reference::To<BasicBuffer> setup_op_input();

    //! Returns a string describing the input type
    string get_input_typestring();

    //! Returns a string describing the output type
    string get_output_typestring();

    //! Initialises the 'op_input' variable
    void initialise_op_input();
    //! Initialises the 'op_output' variable
    void initialise_op_output();

    //! Returns a new Buffer of the requisite type
    Reference::To<BasicBuffer> new_buffer(string typestring);

    //! Called by constructor to initialise stuff
    void init();  
    
    //! The class being run simultaneously
    Reference::To<Operation> op;

    //! True if the thread has been started up
    bool thread_running;

    //! Buffers for storing data    
    vector<Reference::To<BasicBuffer> > input_buffers;
    vector<Reference::To<BasicBuffer> > output_buffers;

    //! Ordered list of who is to be used next
    queue<Reference::To<BasicBuffer> > free_input_buffers;
    queue<Reference::To<BasicBuffer> > full_output_buffers;
    queue<Reference::To<BasicBuffer> > free_input_buffers;
    queue<Reference::To<BasicBuffer> > full_output_buffers;

    //! Mutexes and conds
    pthread_mutex_t free_mutex;
    pthread_cond_t free_cond;
    pthread_mutex_t full_mutex;
    pthread_cond_t full_cond;
    pthread_mutex_t synch_mutex;
    pthread_cond_t synch_cond;

    //! Time-out time on call to pthread_cond_wait (in seconds) [1.0]
    float wait_time;

    //! Whether to use threads [true]
    bool use_threads;

    //! The thread object
    pthread_t* op_thread;
    
    //! Whether the user wants to finish up or keep running [true]
    bool keep_running;

    //! Whether to synchronise work with the main thread [false]
    bool synch;

    //! Has the Operation's input
    Reference::To<BasicBuffer> op_input;
    //! Has the Operation's output
    Reference::To<BasicBuffer> op_output;

  };

}

//! Convenience method to return the Operation
template<class T>
T* dsp::Simultaneous::get(){
  return dynamic_cast<T*>( op.get() );
}

//! Worker function that actually calls 'new' for each element in the vector
template<class T>
bool dsp::Simultaneous::newify_buffers(vector<Reference::To<BasicBuffer> >& buffers){
  for( unsigned i=0; i<buffers.size(); i++)
    buffers[i] = new Buffer<T>;

  return true;
}

#endif
