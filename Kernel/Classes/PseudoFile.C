#include <stdlib.h>

#include "Reference.h"
#include "MJD.h"

#include "dsp/PseudoFile.h"
#include "dsp/File.h"

dsp::PseudoFile::PseudoFile (const dsp::File* f)
{
  Observation::operator = ( *(f->get_info()) );
  filename = f->get_filename();
  header_bytes = f->get_header_bytes();
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

dsp::PseudoFile* dsp::PseudoFile::get_pseudo(string file){
  Reference::To<dsp::File> loader(dsp::File::create(file));
  return new dsp::PseudoFile(loader.get());
}

bool dsp::PseudoFilePtr::operator < (const dsp::PseudoFilePtr& pfp) const{
  if( !ptr || !pfp.ptr )
    throw Error(InvalidState,"dsp::PseudoFilePtr::operator<",
		"One of the PseudoFilePtr's is NULL");

  return this->ptr->operator<(*pfp.ptr);
}
