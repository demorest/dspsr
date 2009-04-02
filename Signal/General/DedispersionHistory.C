/***************************************************************************
 *
 *   Copyright (C) 2005 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/dspExtension.h"
#include "dsp/DedispersionHistory.h"

using namespace std;

//! Null constructor
dsp::DedispersionHistory::DedispersionHistory()
    : dspExtension("DedispersionHistory")
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

//! Add in a dedispersion operation
void
dsp::DedispersionHistory::add(string classname, float dm)
{
    classes.push_back( classname );
    dms.push_back( dm );
}

