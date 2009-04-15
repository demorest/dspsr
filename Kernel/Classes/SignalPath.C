/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SignalPath.h"
#include "dsp/Operation.h"

using namespace std;

dsp::SignalPath::SignalPath (List* list)
  : dspExtension ("SignalPath")
{
  operations = list;
}

void dsp::SignalPath::set_list (List* list)
{
  operations = list;
}

dsp::SignalPath::List* dsp::SignalPath::get_list () const
{
  return operations;
}

dsp::dspExtension* dsp::SignalPath::clone() const
{
  return new SignalPath (*this);
}

void dsp::SignalPath::combine (const SignalPath* that)
{
  if (!operations)
    return;

  if (operations->size() != that->operations->size())
    throw Error (InvalidState, "dsp::UnloaderShare::Storage::combine",
		 "processes have different numbers of operations");

  if (Operation::verbose)
    cerr << "dsp::SignalPath::combine"
      " this=" << this << " that=" << that << endl;

  for (unsigned iop=0; iop < operations->size(); iop++)
  {
    Operation* this_op = (*this->operations)[iop];
    Operation* that_op = (*that->operations)[iop];

    if (Operation::verbose)
      cerr << "dsp::SignalPath::combine " << this_op->get_name() << endl;

    if (this_op->get_name() != that_op->get_name())
      throw Error (InvalidState, "dsp::LoadToFold1::combine",
		   "operation names do not match");

    this_op->combine( that_op );
  }
}

void dsp::SignalPath::reset ()
{
  if (!operations)
    return;

  for (unsigned iop=0; iop < operations->size(); iop++)
    (*operations)[iop]->reset ();
}
