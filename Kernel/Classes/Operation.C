#include <vector>

#include <stdio.h>

#include "Error.h"

#include "dsp/Time.h"
#include "dsp/TimeKeeper.h"
#include "dsp/Operation.h"

//! Global time keeper
dsp::TimeKeeper* dsp::Operation::timekeeper = 0;
 
//! Global flag tells all Operations to record the time spent operating
bool dsp::Operation::record_time = false;

//! Global verbosity flag
bool dsp::Operation::verbose = false;

//! Global instantiation count
int dsp::Operation::instantiation_count = 0;

//! Only ever called by TimeKeeper class
void dsp::Operation::set_timekeeper(TimeKeeper* _timekeeper)
{ timekeeper = _timekeeper; }

void dsp::Operation::unset_timekeeper()
{ timekeeper = 0; }

//! All sub-classes must specify name and capacity for inplace operation
dsp::Operation::Operation (const char* _name)
{
  name = _name;
  id = instantiation_count;
  instantiation_count++;

  discarded_weights = 0;

  optime.operation = "operate";
  
  if( timekeeper )
    timekeeper->add_operation(this);
}

dsp::Operation::~Operation ()
{
  if( timekeeper )
    timekeeper->im_dying(this);
}

void dsp::Operation::operate ()
{
  if (verbose)
    cerr << "dsp::Operation[" << name << "]::operate" << endl;

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

//! Return the number of invalid timesample weights encountered
uint64 dsp::Operation::get_discarded_weights () const
{
  return discarded_weights;
}
 
//! Reset the count of invalid timesample weights encountered
void dsp::Operation::reset_discarded_weights ()
{
  discarded_weights = 0;
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

dsp::Time dsp::Operation::get_operation_time(){
  return Time(optime,name,id);
}

vector<dsp::Time> dsp::Operation::get_extra_times(){
  vector<Time> extra_times(timers.size());

  for(unsigned i=0; i<timers.size();i++)
    extra_times[i] = Time(timers[i],name,id);

  return extra_times;
}

//! Returns the index in the 'timers' array of a particular timer
int dsp::Operation::timers_index(string op_name){
  for( unsigned i=0; i<timers.size(); i++)
    if( timers[i].operation == op_name )
      return i;

  throw Error(InvalidParam,"dsp::Operation::timers_index()",
	      "Your input string '%s' didn't match that of any of the %d timers",
	      op_name.c_str(), timers.size());

  return -1;
}
