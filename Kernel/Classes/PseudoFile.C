/***************************************************************************
 *
 *   Copyright (C) 2003 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <stdlib.h>

#include "Reference.h"
#include "MJD.h"
#include "Error.h"

#include "dsp/File.h"
#include "dsp/dspExtension.h"
#include "dsp/Observation.h"
#include "dsp/PseudoFile.h"

using namespace std;

dsp::PseudoFile::PseudoFile (File* f)
{
  Observation::operator = ( *f->get_info() );
  filename = f->get_filename();
  header_bytes = f->get_header_bytes();
  bs_index = f->get_bs_index();
  //subsize = 0;
}

bool dsp::PseudoFile::operator < (const PseudoFile& in) const
{
  if( get_start_time()==in.get_start_time() ) {
    if( fabs(get_centre_frequency()-in.get_centre_frequency())<0.00001 )
      return false;
    else return fabs(get_centre_frequency()) < fabs(in.get_centre_frequency());
  }

  return get_start_time() < in.get_start_time();
}

dsp::PseudoFile* dsp::PseudoFile::get_pseudo(string file,int bs_index){
  Reference::To<dsp::File> loader(dsp::File::create(file,bs_index));
  return new dsp::PseudoFile(loader.get());
}

bool dsp::PseudoFilePtr::operator < (const dsp::PseudoFilePtr& pfp) const{
  if( !ptr || !pfp.ptr )
    throw Error(InvalidState,"dsp::PseudoFilePtr::operator<",
		"One of the PseudoFilePtr's is NULL");

  return this->ptr->operator<(*pfp.ptr);
}
