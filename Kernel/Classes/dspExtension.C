#include "dsp/dspExtension.h"

//! Constructor
dsp::dspExtension::dspExtension(string _name, bool _can_only_have_one) : Printable(_name) {
  can_only_have_one = _can_only_have_one;
}

//! Virtual destructor
dsp::dspExtension::~dspExtension(){ }
