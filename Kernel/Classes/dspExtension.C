#include "Reference.h"
#include "Printable.h"

#include "dsp/dspExtension.h"

//! Constructor
//dsp::dspExtension::dspExtension(string _name, bool _can_only_have_one) : Printable(_name) {
dsp::dspExtension::dspExtension(string _name, bool _can_only_have_one) : Reference::Able() {
  name = _name;
  can_only_have_one = _can_only_have_one;
}

//! Virtual destructor
dsp::dspExtension::~dspExtension(){ }
