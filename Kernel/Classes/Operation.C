#include <stdio.h>

#include "dsp/TimeKeeper.h"

#include "dsp/Operation.h"
#include "Error.h"

dsp::TimeKeeper* say_wazzup(){
  //fprintf(stderr,"wazzup\n");
  return 0;
}
 
dsp::TimeKeeper* dsp::Operation::timekeeper = say_wazzup();
 
//! Global flag tells all Operations to record the time spent operating
bool dsp::Operation::record_time = false;

//! Global verbosity flag
bool dsp::Operation::verbose = false;

//! Only ever called by TimeKeeper class
void dsp::Operation::set_timekeeper(TimeKeeper* _timekeeper)
{ timekeeper = _timekeeper; }
void dsp::Operation::unset_timekeeper()
{ timekeeper = 0; }

//! All sub-classes must specify name and capacity for inplace operation
dsp::Operation::Operation (const char* _name)
{
  name = _name;

  if( timekeeper )
    timekeeper->add_operation(this);
}

dsp::Operation::~Operation (){
  if( timekeeper )
    timekeeper->im_dying(this);
}

void dsp::Operation::operate ()
{
  if( timekeeper )
    timekeeper->setup_done();

  if (record_time)
    optime.start();

  //! call the pure virtual method defined by sub-classes
  operation();

  if (record_time)
    optime.stop();
}

double dsp::Operation::get_total_time () const
{
  return optime.get_total();
}

double dsp::Operation::get_elapsed_time () const
{
  return optime.get_elapsed();
}

//! Return pointer to a memory resource shared by operations
void* dsp::Operation::workingspace (size_t nbytes)
{
  static char* working_space = NULL;
  static size_t working_size = 0;

  if (!nbytes) {
    if (working_space) delete [] working_space; working_space = 0;
    working_size = 0;
  }

  if (working_size < nbytes) {
    if (working_space) delete [] working_space; working_space = 0;
    working_space = new char [nbytes];

    if (!working_space)
      throw Error (BadAllocation, "Operation::workingspace",
	"error allocating %d bytes",nbytes);

    working_size = nbytes;
  }

  return working_space;
}





