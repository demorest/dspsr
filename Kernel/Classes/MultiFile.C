#include <algorithm>

#include "MultiFile.h"
#include "File.h"
#include "Error.h"

dsp::MultiFile::MultiFile ()
{
  index = 0;
}

//! operator used to sort the File vector
bool operator < (const Reference::To<dsp::File>& f1,
		 const Reference::To<dsp::File>& f2)
{
  return f1->get_info()->get_start_time() < f2->get_info()->get_start_time();
}

void dsp::MultiFile::load (vector<string>& filenames)
{
  if (filenames.empty())
    throw Error (InvalidParam, "dsp::Multifile::load", "no filenames");

  files.resize (filenames.size());

  unsigned ifile;
  int64 total_ndat = 0;

  for (ifile=0; ifile<filenames.size(); ifile++) {
    files[ifile] = File::create (filenames[ifile]);
    total_ndat += files[ifile]->get_info()->get_ndat();
  }

  sort (files.begin(), files.end());

  for (ifile=1; ifile<files.size(); ifile++) {
    
    const Observation* obs1 = files[ifile-1]->get_info();
    const Observation* obs2 = files[ifile]->get_info();

    if (! obs1->contiguous(*obs2))
      throw Error (InvalidParam, "dsp::Multifile::load",
		   "'"+files[ifile-1]->get_filename()+"'"
		   " is not contiguous with "
		   "'"+files[ifile]->get_filename()+"'");

  }

  info = *(files[0]->get_info());
  info.set_ndat (total_ndat);
}

//! Load bytes from file
int64 dsp::MultiFile::load_bytes (unsigned char* buffer, uint64 bytes)
{
  if (verbose)
    cerr << "MultiFile::load_bytes nbytes=" << bytes << endl;

  uint64 bytes_loaded = 0;

  while (bytes_loaded < bytes) {

    int64 to_load = bytes - bytes_loaded;

    if (index >= files.size()) {
      if (verbose)
	cerr << "MultiFile::load_bytes end of data" << endl;
      end_of_data = true;
      break;
    }

    int64 did_load = files[index]->load_bytes (buffer, to_load);

    if (did_load < 0)
      return -1;

    if (did_load < to_load)
      // this File has reached the end of data
      index ++;

    bytes_loaded += did_load;
    buffer += did_load;
  }

  return bytes_loaded;
}

//! Adjust the file pointer
int64 dsp::MultiFile::seek_bytes (uint64 bytes)
{
  if (verbose)
    cerr << "MultiFile::seek_bytes nbytes=" << bytes << endl;

  uint64 total_bytes = 0;

  for (index = 0; index < files.size(); index++) {

    uint64 file_bytes = files[index]->get_info()->nbytes();

    if (bytes < total_bytes + file_bytes)
      break;

    total_bytes += file_bytes;
  }

  if (index == files.size()) {
    cerr << "dsp::MultiFile::seek_bytes (" << bytes << ")"
      " past end of data" << endl;
    return -1;
  }

  int64 seeked = files[index]->seek_bytes (bytes-total_bytes);
  if (seeked < 0)
    return -1;

  return total_bytes + seeked;
}
