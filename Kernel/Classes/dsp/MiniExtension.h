//-*-C++-*-

#ifndef __dsp_MiniExtension_h_
#define __dsp_MiniExtension_h_

#include "Reference.h"
#include "environ.h"

#include "dsp/MiniPlan.h"
#include "dsp/dspExtension.h"

namespace dsp {

  class MiniExtension : public dspExtension {

  public:

    //! Constructor
    MiniExtension();

    //! Copy constructor
    MiniExtension(const MiniExtension& m) : dspExtension("MiniExtension") { copy(m); }

    //! Virtual destructor
    virtual ~MiniExtension();

    //! Copies stuff
    virtual void copy(const MiniExtension& m);

    //! Return a new copy-constructed instance identical to this instance
    MiniExtension* clone() const;

    //! Return a new null-constructed instance
    MiniExtension* new_extension() const{ return new MiniExtension; }

    //! Returns a pointer to the MiniPlan
    MiniPlan* get_miniplan(){ return miniplan.ptr(); }

    //! Returns a pointer to the MiniPlan
    const MiniPlan* get_miniplan() const { return miniplan.ptr(); }

    //! Set the MiniPlan
    void set_miniplan(MiniPlan* _miniplan){ miniplan = _miniplan; }

    //! Returns the subsize
    uint64 get_subsize() const { return subsize; }

    //! Sets the subsize
    void set_subsize(uint64 _subsize){ subsize = _subsize; }

    //! Dump out to a string
    virtual string dump_string() const;

  protected:

    //! Pointer to the MiniPlan
    Reference::To<MiniPlan> miniplan;

    //! The number of bytes each chan/pol uses
    uint64 subsize;

  };

}

#endif
