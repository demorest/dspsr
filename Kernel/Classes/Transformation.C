/***************************************************************************
 *
 *   Copyright (C) 2005 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/Transformation.h"

dsp::TransformationBase::DefaultBufferingPolicy dsp::TransformationBase::default_buffering_policy = dsp::TransformationBase::no_buffering_policy;

dsp::TransformationBase::~TransformationBase(){ }
