#include <vector>

#include <stdio.h>

#include "Error.h"

#include "dsp/Time.h"
#include "dsp/TimeKeeper.h"
#include "dsp/Operation.h"


dsp::TimeKeeper* say_wazzup(){
  //fprintf(stderr,"wazzup\n");
  return 0;
}
 
dsp::TimeKeeper* dsp::Operation::timekeeper = say_wazzup();
 
//! Global flag tells all Operations to record the time spent operating
bool dsp::Operation::record_time = false;

//! Global verbosity flag
bool dsp::Operation::verbose = false;

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

  //  fprintf(stderr,"WAZOO\tname='%s' id=%d\n",
  //  name.c_str(),id);

  optime.operation = "operate";
  
  if( timekeeper )
    timekeeper->add_operation(this);

}

dsp::Operation::~Operation (){
  if( timekeeper )
    timekeeper->im_dying(this);
}

void dsp::Operation::operate ()
{
  //fprintf(stderr,"dsp::Operation::operate() name=%s\n",name.c_str());
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

dsp::Time dsp::Operation::get_operation_time(){
  return Time(optime,name,id);
}

vector<dsp::Time> dsp::Operation::get_extra_times(){
  vector<Time> extra_times(timers.size());

  for(unsigned i=0; i<timers.size();i++)
    extra_times[i] = Time(timers[i],name,id);

  return extra_times;
}

void dsp::Operation::cease_timing(){
  for(unsigned i=0; i<timers.size();i++)
    timers[i].stop();
}

