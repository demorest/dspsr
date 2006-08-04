//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Attic/SetBufferingPolicy.h,v $
   $Revision: 1.1 $
   $Date: 2006/08/04 00:08:09 $
   $Author: straten $ */

#ifndef __dspsr_SetBufferingPolicy_h
#define __dspsr_SetBufferingPolicy_h

#include "dsp/Transformation.h"

namespace dsp {

  class SetBufferingPolicy {

  public:

    typedef enum { None, Input, Output } Policy;

    static Policy policy;

    static void set (TransformationBase* base);

  };

}

#endif
