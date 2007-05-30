/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SetBufferingPolicy.h"
#include "dsp/TimeSeries.h"
#include "dsp/InputBuffering.h"
#include "dsp/OutputBuffering.h"

using namespace std;

dsp::SetBufferingPolicy::Policy dsp::SetBufferingPolicy::policy =
  dsp::SetBufferingPolicy::None;

void dsp::SetBufferingPolicy::set (TransformationBase* base)
{
  //cerr << "dsp::SetBufferingPolicy::set" << endl;

  if (policy == Output) {

    //cerr << "dsp::SetBufferingPolicy::set policy == Output" << endl;

    HasOutput<TimeSeries>* tr = dynamic_cast<HasOutput<TimeSeries>*>(base);

    if (tr && base->get_type() != inplace)
      base->set_buffering_policy( new OutputBuffering(tr) );

  }

  if (policy == Input) {

    //cerr << "dsp::SetBufferingPolicy::set policy == Input" << endl;

    HasInput<TimeSeries>* tr = dynamic_cast<HasInput<TimeSeries>*>(base);

    if (tr && base->get_type() != inplace)
      base->set_buffering_policy( new InputBuffering(tr) );

  }
}
