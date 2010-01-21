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

/*! By default, operations do not time themselves */
bool dsp::Operation::record_time = false;

/*! By default, if record_time is enabled, operations will report total
  time spent when their destructor is called */
bool dsp::Operation::report_time = true;

//! Global verbosity flag
bool dsp::Operation::verbose = false;

//! Global instantiation count
int dsp::Operation::instantiation_count = 0;

bool dsp::Operation::check_state = true;

int dsp::Operation::operation_status = 0;

dsp::Operation::Operation (const Operation& op)
{
  scratch = op.scratch;
  set_scratch_called = op.set_scratch_called;
  name = op.name;
  prepared = false;

  id = instantiation_count;
  instantiation_count++;

  discarded_weights = op.discarded_weights;
  total_weights = op.total_weights;
}

//! All sub-classes must specify name and capacity for inplace operation
dsp::Operation::Operation (const char* _name)
{
  if (_name)
    name = _name;
  else
    name = "Operation";

  id = instantiation_count;
  instantiation_count++;

  discarded_weights = 0;
  total_weights = 0;

  scratch = Scratch::get_default_scratch();
  set_scratch_called = false;

  prepared = false;
}


dsp::Operation::~Operation ()
{
  if (report_time)
    Operation::report ();
}

bool dsp::Operation::can_operate()
{
  return true;
}

void dsp::Operation::prepare ()
{
  prepared = true;
}

void dsp::Operation::reserve ()
{
}

void dsp::Operation::add_extensions (Extensions*)
{
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
uint64_t dsp::Operation::get_discarded_weights () const
{
  return discarded_weights;
}

//! Return the number of invalid timesample weights encountered
uint64_t dsp::Operation::get_total_weights () const
{
  return total_weights;
}

void dsp::Operation::set_scratch (Scratch* s)
{
  scratch = s;
  set_scratch_called = true;
}

bool dsp::Operation::scratch_was_set () const
{
  return set_scratch_called;
}

//! Combine accumulated results with another operation
void dsp::Operation::combine (const Operation* other)
{
  if (this == other)
    return;

  total_weights += other->total_weights;
  discarded_weights += other->discarded_weights;

  optime += other->optime;
}

//! Reset accumulated results to zero
void dsp::Operation::reset ()
{
  discarded_weights = 0;
  total_weights = 0;
}

//! Report operation statistics
void dsp::Operation::report () const
{
  if (!record_time || !get_total_time())
    return;

  unsigned cwidth = 25;

  static bool first_report = true;

  if (first_report)
  {
    cerr << pad (cwidth, "Operation")
	 << pad (cwidth, "Time Spent")
	 << pad (cwidth, "Discarded") << endl;

    first_report = false;
  }

  cerr << pad (cwidth, get_name())
       << pad (cwidth, tostring(get_total_time()))
       << pad (cwidth, tostring(get_discarded_weights()))
       << endl;
}
