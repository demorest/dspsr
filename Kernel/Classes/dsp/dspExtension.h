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
    dspExtension (std::string _name, bool _can_only_have_one=true);

    //! Virtual destructor
    virtual ~dspExtension();

    //! Return a new copy-constructed instance identical to this instance
    virtual dspExtension* clone() const = 0;

    //! Return a new null-constructed instance
    virtual dspExtension* new_extension() const = 0;

    //! If true, then you can only have one of this type of dspExtension per Observation instantiation
    virtual bool must_only_have_one() const;

    //! Dump out to a string
    virtual std::string dump_string() const = 0;

    //! Delete this if dspExtension inherits from Printable
    std::string get_name() const { return name; }

  private:

    //! Delete this if dspExtension inherits from Printable
    std::string name;

    //! If true, then you can only have one of this type of dspExtension per Observation instantiation
    bool can_only_have_one;

  };

}

#endif
