//-*-C++-*-

#ifndef __dsp_dspExtension_h_
#define __dsp_dspExtension_h_

#include <string>

#include "Printable.h"
#include "Reference.h"

namespace dsp{

  // Printable has a 'name' attribute
  //  class dspExtension : public Printable {
  class dspExtension : public Reference::Able {

  public:

    //! Constructor
    dspExtension(string _name, bool _can_only_have_one=true);

    //! Virtual destructor
    virtual ~dspExtension();

    //! Return a new copy-constructed instance identical to this instance
    virtual dspExtension* clone() const = 0;

    //! Return a new null-constructed instance
    virtual dspExtension* new_extension() const = 0;

    //! If true, then you can only have one of this type of dspExtension per Observation instantiation
    bool must_only_have_one() const { return can_only_have_one; }

    //! Delete this if dspExtension inherits from Printable
    string get_name() const { return name; }

  private:

    //! Delete this if dspExtension inherits from Printable
    string name;

    //! If true, then you can only have one of this type of dspExtension per Observation instantiation
    bool can_only_have_one;

  };

}

#endif
