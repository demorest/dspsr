#include "dsp/PseudoFile.h"
#include "dsp/File.h"
#include "dsp/Observation.h"

dsp::PseudoFile::PseudoFile(const File& f){
  Observation::operator=(*f.get_info());
  filename=f.get_filename();
  header_bytes=f.get_header_bytes();
}

bool dsp::PseudoFile::operator < (const PseudoFile& in) const{
  if( get_start_time()==in.get_start_time() ){
    if( fabs(get_centre_frequency()-in.get_centre_frequency())<0.00001 )
      return false;
    else return fabs(get_centre_frequency()) < fabs(in.get_centre_frequency());
  }

  return get_start_time() < in.get_start_time();
}
