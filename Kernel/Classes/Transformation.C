#include "dsp/Transformation.h"

dsp::TransformationBase::DefaultBufferingPolicy dsp::TransformationBase::default_buffering_policy = dsp::TransformationBase::no_buffering_policy;

dsp::TransformationBase::~TransformationBase(){ }
