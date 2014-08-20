//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2014 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __OutputFileShare_h
#define __OutputFileShare_h

#include "Error.h"
#include "MJD.h"
#include "dsp/HasInput.h"
#include "dsp/Operation.h"
#include "dsp/BitSeries.h"
#include "dsp/OutputFile.h"

class ThreadContext;

namespace dsp {

  //! Share one OutputFile among multiple processing threads 
  /*! Coordinates writing of BitSeries output from multiple threads into a
   * single, correctly time-ordered file.  
   *
   * The algorithm is for each thread to wait its turn, only writing data when
   * it has the expected next piece.  If all threads are waiting, the one with
   * the earliest start time is written (should not happen for contiguous
   * data).
   *
   * Some elements of this class are patterned after UnloaderShare and could
   * potentially be refactored into a common base class.
   */
  class OutputFileShare : public Reference::Able { public:
    
    //! Constructor
    OutputFileShare (unsigned contributors);
    
    //! Destructor
    virtual ~OutputFileShare ();

    //! Get the output file
    OutputFile* get_output_file () const { return output_file; }

    //! Set the output file
    void set_output_file (OutputFile* f) { output_file = f; }

    //! Set the thread context
    void set_context (ThreadContext* t) { context = t; }

    //! Set the thread context
    ThreadContext* get_context () const { return context; }

    //! Class used to signal when each contributor is ready
    class Submit;

    //! Return Submit interface for given contributor
    Submit* new_Submit (unsigned contributor);

    //! Return next time
    MJD get_next_time () const { return next_time; }

    //! set next time
    void set_next_time (MJD time) { next_time=time; }

    //! Call when a contributor is ready to write
    void signal_ready (unsigned contributor, MJD start_time);

    //! Call when a contributor is done
    void signal_done (unsigned contributor);

  protected:

    //! Number of contributors
    unsigned contributors;

    //! The actual output file unloader
    Reference::To<OutputFile> output_file;

    //! Coordinate various threads
    ThreadContext* context;

    //! Leading edge time of next output sample
    MJD next_time;

    //! Current start times of waiting contributors
    std::vector<MJD> start_times;

    //! first-time-through flag
    bool first;

    //! Number of threads currently ready to write data
    unsigned nready;

  };

  class OutputFileShare::Submit : public OutputFile
  {
  public:

    //! Constructor
    Submit (OutputFileShare* parent, unsigned contributor);

    //! wrapper for file write operation
    void operation ();

    //! Get the filename extension
    std::string get_extension () const 
    { 
      return parent->output_file->get_extension();
    };

    //! Write the header
    void write_header () { parent->output_file->write_header(); }

  protected:

    Reference::To<OutputFileShare> parent;
    unsigned contributor;

  };

}

#endif // !defined(__OutputFileShare_h)
