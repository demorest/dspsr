/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Operation.h"
#include "dsp/TimeKeeper.h"
#include "dsp/Scratch.h"

using namespace std;

//! Global time keeper
dsp::TimeKeeper* dsp::Operation::timekeeper = 0;

//! If this is set to true then dsp::Transformation nukes prepends with test values
bool dsp::Operation::debug = false;

//! Global flag tells all Operations to record the time spent operating
bool dsp::Operation::record_time = false;

//! Global verbosity flag
bool dsp::Operation::verbose = false;

//! Global instantiation count
int dsp::Operation::instantiation_count = 0;

bool dsp::Operation::check_state = true;

/*! No buffering policy is enabled by default.  If input buffering is
  desired, this flag must be raised before any Operation-derived
  classes are instantiated */
bool dsp::Operation::preserve_data = false;

int dsp::Operation::operation_status = 0;

//! Only ever called by TimeKeeper class
void dsp::Operation::set_timekeeper(TimeKeeper* _timekeeper)
{ timekeeper = _timekeeper; }

void dsp::Operation::unset_timekeeper()
{ timekeeper = 0; }

//! Set verbosity ostream
void dsp::Operation::set_ostream (std::ostream& os)
{
  this->cerr.rdbuf( os.rdbuf() );
}

dsp::Operation::Operation (const Operation& op)
  : cerr (op.cerr.rdbuf())
{
  scratch = op.scratch;
  name = op.name;
  prepared = false;
}

//! All sub-classes must specify name and capacity for inplace operation
dsp::Operation::Operation (const char* _name)
  : cerr (std::cerr.rdbuf())
{
  if (_name)
    name = _name;
  else
    name = "Operation";

  id = instantiation_count;
  instantiation_count++;

  discarded_weights = 0;

  optime.operation = "operate";

  scratch = &Scratch::default_scratch;

  if( timekeeper )
    timekeeper->add_operation(this);

  prepared = false;
  // set_ostream (std::cerr);
}

dsp::Operation::~Operation ()
{
  if( timekeeper && record_time )
    timekeeper->im_dying(this);
}

bool dsp::Operation::can_operate(){
  return true;
}

void dsp::Operation::prepare ()
{
  prepared = true;
}

bool dsp::Operation::operate ()
{
  if (verbose)
    cerr << "dsp::Operation[" << name << "]::operate" << endl;

  if( timekeeper )
    timekeeper->setup_done();

  if (record_time)
    optime.start();

  if( !can_operate() )
    return false;

  operation_status = 0;

  //! call the pure virtual method defined by sub-classes
  operation();

  if (record_time)
    optime.stop();

  if( operation_status != 0 )
    return false;

  return true;
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
int dsp::Operation::timers_index(const string& op_name){
  for( unsigned i=0; i<timers.size(); i++)
    if( timers[i].operation == op_name )
      return i;

  throw Error(InvalidParam,"dsp::Operation::timers_index()",
	      "Your input string '%s' didn't match that of any of the %d timers",
	      op_name.c_str(), timers.size());

  return -1;
}

void dsp::Operation::set_scratch (Scratch* s)
{
  scratch = s;
}
