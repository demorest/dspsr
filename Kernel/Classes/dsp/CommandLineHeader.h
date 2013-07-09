//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2012 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __CommandLineHeader_h
#define __CommandLineHeader_h

#include "Reference.h"

namespace dsp {

  //! Allow ascii_header-style params to be entered on the command line
  class CommandLineHeader : public Reference::Able
  {

  public:
    //! Parse the given argc, argv into a header file, return file name
    std::string convert(int argc, char **argv, std::string filename="");

  };

}

#endif // !defined(__CommandLineHeader_h)
