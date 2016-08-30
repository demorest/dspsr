/***************************************************************************
 *
 *   Copyright (C) 2014 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/OutputFileShare.h"
#include "dsp/OutputFile.h"
#include "dsp/BitSeries.h"

#include "Error.h"
#include "ThreadContext.h"

#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

#include <algorithm>

using namespace std;

//! Constructor
dsp::OutputFileShare::OutputFileShare (unsigned _contributors)
{
  contributors = _contributors;
  context = 0;
  next_time = 0.;
  first = true;
  nready = 0;
  start_times.resize(contributors);
  for (unsigned i=0; i<contributors; i++) 
    start_times[i]=0.0;
}
    
//! Destructor
dsp::OutputFileShare::~OutputFileShare ()
{
}

dsp::OutputFileShare::Submit*
dsp::OutputFileShare::new_Submit (unsigned contributor)
{
  return new Submit(this, contributor);
}

//! Signal that a thread is ready to write
// Note, this should be called with the context lock already acquired
void dsp::OutputFileShare::signal_ready (unsigned contributor, MJD start_time)
{

  nready++;
  start_times[contributor] = start_time;

  if (Operation::verbose)
    cerr << "dsp::OutputFileShare::signal_ready"
      << " contributor=" << contributor
      << " nready=" << nready
      << " contributors=" << contributors
      << endl;

  // If all threads are ready, adjust the start time to match the earliest.
  // Normally this should only happen the first time through.  If this
  // condition occurs at other times, this could indicate non-contiguous
  // data.
  if (nready==contributors)
  {
    MJD min_time = *min_element(start_times.begin(),start_times.end());

    if (Operation::verbose)
      cerr << "dsp::OutputFileShare::signal_ready all threads ready"
        << " start_time=" << next_time
        << endl;

    if (first==false && (min_time!=next_time)) 
      cerr << "dsp::OutputFileShare::signal_ready missing data! (diff=" 
        << (min_time-next_time).in_seconds() << "s)"
        << endl;

    next_time = min_time;

    first = false;

    // Wake up all waiting threads
    context->broadcast();
  }

}

//! Signal that a thread is done
void dsp::OutputFileShare::signal_done (unsigned contributor)
{
  // Note, locking is not used here since the thread should already
  // have the lock when calling this.
  nready--;
  start_times[contributor] = 0.0;
}

dsp::OutputFileShare::Submit::Submit (OutputFileShare* _parent, 
    unsigned _contributor) : OutputFile("OutputFileShare")
{
  parent = _parent;
  contributor = _contributor;
}

void dsp::OutputFileShare::Submit::operation ()
{

  ThreadContext::Lock lock(parent->get_context());
  parent->signal_ready(contributor, get_input()->get_start_time());

  if (Operation::verbose) 
    cerr << "dsp::OutputFileShare::Submit:operation " << contributor 
      << " mjd0=" << get_input()->get_start_time()
      << " mjd1=" << get_input()->get_end_time()
      << " want=" << parent->get_next_time()
      << endl;

  while (get_input()->get_start_time() > parent->get_next_time())
  {
    // Wait
    //cerr << "OFS " << contributor << " waiting" << endl;
    // if input has no data, don't block
    if (!get_input()->get_ndat())
      break;
    parent->get_context()->wait();
  }

  if (get_input()->get_start_time() < parent->get_next_time())
  {
    // Data are out of order, discard.  This shouldn't happen...
    cerr << "dsp::OutputFileShare::Submit::operation" 
      << " contributor=" << contributor << " discarding misordered data!" 
      << endl;
  }

  else if (get_input()->get_start_time() == parent->get_next_time())
  {
    // Write data
    parent->get_output_file()->set_input(get_input());
    parent->get_output_file()->operation();

    // update next_sample
    parent->set_next_time(input->get_end_time());
  }

  // Mark thread as done
  parent->signal_done(contributor);

  // Signal waiting threads to wake up
  parent->get_context()->broadcast();

}
