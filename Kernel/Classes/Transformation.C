#include <string>

#include "psr_cpp.h"

#include "dsp/Operation.h"
#include "dsp/Transformation.h"

dsp::TransformationBase::TransformationBase(const char* _name)
  : Operation(_name){ }

dsp::TransformationBase::~TransformationBase(){ }
