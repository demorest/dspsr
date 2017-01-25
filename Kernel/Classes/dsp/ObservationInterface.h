//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/ObservationInterface.h

#ifndef __dsp_ObservationTI_h
#define __dsp_ObservationTI_h

#include "dsp/Observation.h"
#include "TextInterface.h"

namespace dsp
{

  //! Provides a text interface to get and set Observation attributes
  class Observation::Interface : public TextInterface::To<Observation>
  {

  public:

    //! Default constructor that takes an optional instance
    Interface ( Observation* = 0 );

    //! Set the instance to which this interface interfaces
    void set_instance (Observation*) ;

    //! clone this text interface
    TextInterface::Parser *clone();
    
    //! Get the interface name
    std::string get_interface_name() const { return "Observation::Interface"; }

  };

}


#endif
