//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Dump_h_
#define __dsp_Dump_h_

#include "dsp/Sink.h"
#include "dsp/TimeSeries.h"

#include "FilePtr.h"

namespace dsp {

  //! Dumps TimeSeries data to file in either binary or ascii format
  /*! This diagnostic tool can be inserted into a signal path to have
    a look at what is going in and out of and on in. */
  class Dump : public Sink<TimeSeries>
  {
  public:

    //! Null constructor
    Dump (const char* name = "Dump");
    ~Dump ();

    //! Set the output stream to which data will be dumped
    void set_output (FILE*);

    //! Set the flag to output binary data
    void set_output_binary (bool flag=true);

    //! In binary mode, write an ASCII (DADA) header
    void prepare ();

    Operation::Function get_function () const { return Operation::Structural; }

  protected:

    //! Write to the open file descriptor
    void calculation ();

    FilePtr output;
    bool binary;
  };

}

#endif

