#include "dsp/Operation.h"
#include "Error.h"

//! Global flag tells all Operations to record the time spent operating
bool dsp::Operation::record_time = false;

//! Global verbosity flag
bool dsp::Operation::verbose = false;

//! All sub-classes must specify name and capacity for inplace operation
dsp::Operation::Operation (const char* _name)
{
  name = _name;
}

dsp::Operation::~Operation ()
{
}

void dsp::Operation::operate ()
{
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
      throw Error (BadAlloc, "Operation::workingspace",
	"error allocating %d bytes",nbytes);

    working_size = nbytes;
  }

  return working_space;
}

