/***************************************************************************
 *
 *   Copyright (C) 2005 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <string>

#include "string_utils.h"
#include "psr_cpp.h"
#include "format_it.h"

#include "dsp/dspExtension.h"
#include "dsp/DedispersionHistory.h"

//! Null constructor
dsp::DedispersionHistory::DedispersionHistory()
    : dspExtension("DedispersionHistory",true)
{
}

//! Virtual destructor
dsp::DedispersionHistory::~DedispersionHistory()
{
}

//! Return a new copy-constructed instance identical to this instance
dsp::dspExtension*
dsp::DedispersionHistory::clone() const
{
    DedispersionHistory* dh = new DedispersionHistory;
    dh->classes = classes;
    dh->dms = dms;

    return dh;
}

//! Return a new null-constructed instance
dsp::dspExtension*
dsp::DedispersionHistory::new_extension() const
{
    return new DedispersionHistory;
}

//! Add in a dedispersion operation
void
dsp::DedispersionHistory::add(string classname, float dm)
{
    classes.push_back( classname );
    dms.push_back( dm );
}

//! Dump out to a string
string
dsp::DedispersionHistory::dump_string() const
{
  vector<string> lines;

  for( unsigned i=0; i<classes.size(); i++)
    lines.push_back(classes[i] + " " + make_string(dms[i]) + "\n");

  format_it(lines,3);

  string s;
  for( unsigned i=0; i<classes.size(); i++)
    s += lines[i];

  return s;
}
