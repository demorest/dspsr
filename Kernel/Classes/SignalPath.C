/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SignalPath.h"
#include "dsp/Operation.h"

using namespace std;

dsp::SignalPath::SignalPath (const List& _list)
  : dspExtension ("SignalPath")
{
  list = _list;
}

dsp::SignalPath::SignalPath (const vector< Reference::To<Operation> >& _list)
  : dspExtension ("SignalPath")
{
  list.resize (_list.size());
  for (unsigned i=0; i<list.size(); i++)
    list[i] = _list[i];
}

void dsp::SignalPath::set_list (const List& _list)
{
  list = _list;
}

const dsp::SignalPath::List* dsp::SignalPath::get_list () const
{
  return &list;
}

dsp::dspExtension* dsp::SignalPath::clone() const
{
  return new SignalPath (*this);
}

void dsp::SignalPath::add (Operation* op)
{
  list.push_back (op);
}

void dsp::SignalPath::combine (const SignalPath* that)
{
  if (list.size() != that->list.size())
    throw Error (InvalidState, "dsp::UnloaderShare::Storage::combine",
		 "processes have different numbers of operations");

  if (Operation::verbose)
    cerr << "dsp::SignalPath::combine"
      " this=" << this << " that=" << that << endl;

  for (unsigned iop=0; iop < list.size(); iop++)
  {
    Operation* this_op = this->list[iop];
    Operation* that_op = that->list[iop];

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
  for (unsigned iop=0; iop < list.size(); iop++)
    list[iop]->reset ();
}
