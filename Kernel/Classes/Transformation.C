#include "dsp/Transformation.h"

bool dsp::TransformationBase::check_state = true;

dsp::TransformationBase::TransformationBase(const char* _name)
  : Operation(_name)
{ 
}

dsp::TransformationBase::~TransformationBase(){ }
