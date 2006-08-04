#include "dsp/SetBufferingPolicy.h"
#include "dsp/TimeSeries.h"
#include "dsp/InputBuffering.h"
#include "dsp/OutputBuffering.h"

dsp::SetBufferingPolicy::Policy dsp::SetBufferingPolicy::policy;

void dsp::SetBufferingPolicy::set (TransformationBase* base)
{
  if (policy == Output) {

    HasOutput<TimeSeries>* tr = dynamic_cast<HasOutput<TimeSeries>*>(base);

    if (tr && tr->get_type() != inplace)
      tr->set_buffering_policy( new OutputBuffering(tr) );

  }

  if (policy == Input) {

    HasInput<TimeSeries>* tr = dynamic_cast<HasInput<TimeSeries>*>(base);

    if (tr && tr->get_type() != inplace)
      tr->set_buffering_policy( new InputBuffering(tr) );

  }
}
