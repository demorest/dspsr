/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Operation.h"
#include "dsp/Scratch.h"
#include "strutil.h"

using namespace std;

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

//! Set verbosity ostream
void dsp::Operation::set_ostream (std::ostream& os) const
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

  scratch = Scratch::get_default_scratch();

  prepared = false;
}

static bool first_destructor = true;

dsp::Operation::~Operation ()
{
  if (!record_time || !get_total_time())
    return;

  unsigned cwidth = 25;

  if (first_destructor)
  {
    cerr << pad (cwidth, "Operation")
	 << pad (cwidth, "Time Spent")
	 << pad (cwidth, "Discarded") << endl;

    first_destructor = false;
  }

  cerr << pad (cwidth, get_name())
       << pad (cwidth, tostring(get_total_time()))
       << pad (cwidth, tostring(get_discarded_weights()))
       << endl;
}

bool dsp::Operation::can_operate()
{
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

void dsp::Operation::set_scratch (Scratch* s)
{
  scratch = s;
}

