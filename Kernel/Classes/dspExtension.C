/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/dspExtension.h"
#include "typeutil.h"

using namespace std;

//! Constructor
dsp::dspExtension::dspExtension (const string& _name)
{
  name = _name;
}


//! Adds a dspExtension
void dsp::Extensions::add_extension (dspExtension* ext)
{
  unsigned index = find (extension, ext);

  if (index < extension.size())
    extension[index] = ext;
  else
    extension.push_back (ext);
}

//! Returns the number of dspExtensions currently stored
unsigned dsp::Extensions::get_nextension() const
{
  return extension.size();
}

//! Returns the i'th dspExtension stored
dsp::dspExtension* dsp::Extensions::get_extension(unsigned iext){
  if( iext >= extension.size() )
    throw Error(InvalidParam,"dsp::Extensions::get_extension()",
              "You requested extension '%d' but there are only %d extensions stored",iext,extension.size());
  return extension[iext].get();
}

//! Returns the i'th dspExtension stored
const dsp::dspExtension* dsp::Extensions::get_extension(unsigned iext) const{
  if( iext >= extension.size() )
    throw Error(InvalidParam,"dsp::Extensions::get_extension()",
              "You requested extension '%d' but there are only %d extensions stored",iext,extension.size());
  return extension[iext].get();
}
