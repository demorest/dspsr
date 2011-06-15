#include "dsp/PlotFactory.h"

#include "dsp/HistoPlot.h"
#include "dsp/FrequencyVsTime.h"

#include "Error.h"
#include "pad.h"

dsp::PlotFactory::PlotFactory()
{
  add (new PlotFactory::Advocate<HistoPlot>
      ('H', "hist", "Histogram of samples") );

  add (new PlotFactory::Advocate<FrequencyVsTime>
      ('F', "freq", "Frequency vs time") );
}

dsp::Plot* dsp::PlotFactory::construct(string name)
{
  char shortcut = 0;

  if (name.length() == 1)
    shortcut = name[0];

  for (unsigned i=0; i < agents.size(); i++) {
    if ((shortcut && shortcut == agents[i]->get_shortcut()) ||
        (name == agents[i]->get_name())) {
      return agents[i]->construct();
    }
  }

  throw Error (InvalidParam, "Pulsar::PlotFactory::construct",
      "no Plot named " + name);
}

string dsp::PlotFactory::get_name(const unsigned i)
{
  return agents[i]->get_name();
}

string dsp::PlotFactory::get_description(const unsigned i)
{
  return agents[i]->get_description();
}

string dsp::PlotFactory::help()
{
  unsigned ia, maxlen = 0;
  for (ia=0; ia < agents.size(); ia++) {
    if (get_name(ia).length() > maxlen) {
      maxlen = get_name(ia).length();
    }
  }

  maxlen += 2;
  string result;

  for (ia=0; ia < agents.size(); ia++) {
    result += pad(maxlen, get_name(ia)) +
      "[" +
      agents[ia]->get_shortcut() +
      "]  " +
      get_description(ia) +
      "\n";
  }

  return result;
}



