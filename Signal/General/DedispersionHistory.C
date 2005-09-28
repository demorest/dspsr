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
    return new DedipersionHistory;
}

//! Add in a dedispersion operation
void
dsp::DedispersionHistory::add(string classname, float dm)
{
    classes.push_back( classname );
    dms.push_back( dm );
}



