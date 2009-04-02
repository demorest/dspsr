//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_dspExtension_h_
#define __dsp_dspExtension_h_

#include "Reference.h"
#include <string>

namespace dsp{

  // Printable has a 'name' attribute
  class dspExtension : public Reference::Able {

  public:

    //! Constructor
    dspExtension (const std::string& name);

    //! Return a new copy-constructed instance identical to this instance
    virtual dspExtension* clone() const = 0;

    //! Delete this if dspExtension inherits from Printable
    std::string get_name() const { return name; }

  private:

    //! Delete this if dspExtension inherits from Printable
    std::string name;

  };

}

#endif
