#include <algorithm>

#include <math.h>

#include "genutil.h"

#include "Error.h"

#include "dsp/File.h"

#include "dsp/MultiFile.h"

dsp::MultiFile::MultiFile () : Seekable ("MultiFile")
{
  index = 0;
}

//! operator used to sort the File vector
bool operator < (const Reference::To<dsp::File>& f1,
		 const Reference::To<dsp::File>& f2)
{
  return f1->get_info()->get_start_time() < f2->get_info()->get_start_time();
}

void dsp::MultiFile::open (vector<string> filenames)
{
  if (filenames.empty())
    throw Error (InvalidParam, "dsp::Multifile::open()", "An empty list of filenames has been given to this method");

  vector<string> old_filenames;
  unsigned ifile;

  int64 total_ndat = 0;

  for( ifile=0; ifile<files.size(); ifile++){
    old_filenames.push_back( files[ifile]->get_filename() );
    total_ndat += files[ifile]->get_info()->get_ndat();
  }

  for( unsigned i=0; i<filenames.size(); i++){
    if( !is_one_of(filenames[i],old_filenames) ){
      files.push_back( File::create(filenames[i]) );
      old_filenames.push_back( filenames[i] );
      total_ndat += files.back()->get_info()->get_ndat();
    }
  }

  ensure_contiguity();

  info = *(files[0]->get_info());
  info.set_ndat (total_ndat);

  reset();
}

//! Makes sure only these filenames are open
void dsp::MultiFile::have_open(vector<string> filenames){

  // Erase any files we already have open that we don't want open
  for( unsigned ifile=0; ifile<files.size(); ifile++){
    if( !is_one_of(files[ifile]->get_filename(),filenames) ){
      files.erase(files.begin()+ifile);
      ifile--;
    }
  }

  // Make a list of files we still have open
  vector<string> old_filenames; 
  for( unsigned ifile=0; ifile<files.size(); ifile++)
    old_filenames.push_back( files[ifile]->get_filename() );

  // Open any files we don't already have open
  for( unsigned i=0; i<filenames.size(); i++)
    if( !is_one_of(filenames[i],old_filenames) )
      files.push_back( File::create(filenames[i]) );

  // sort and ensure contiguity
  ensure_contiguity();

  // work out the total ndat
  int64 total_ndat = 0;
  for( unsigned ifile=0; ifile<files.size(); ifile++)
    total_ndat += files[ifile]->get_info()->get_ndat();

  // set up info
  info = *(files[0]->get_info());
  info.set_ndat (total_ndat);

  // reset those file pointers
  reset();
}

//! Erase the entire list of loadable files
void dsp::MultiFile::erase_files(){
  files.erase( files.begin(), files.end());
  info = Observation();
  reset();
}

//! Erase just some of the list of loadable files
void dsp::MultiFile::erase_files(vector<string> erase_filenames){
  for( unsigned ifile=0; ifile<files.size(); ifile++){
    if( is_one_of(files[ifile]->get_filename(),erase_filenames) ){
      files.erase( files.begin()+ifile );
      ifile--;
    }
  }
  
  if( files.empty() ){
    info = Observation();
    reset();
    return;
  }

  ensure_contiguity();

  uint64 total_ndat = 0;
  
  for( unsigned ifile=0; ifile<files.size(); ifile++)
    total_ndat += files[ifile]->get_info()->get_ndat();

  info = *(files[0]->get_info());
  info.set_ndat (total_ndat);

  reset();
}

void dsp::MultiFile::ensure_contiguity(){
  sort( files.begin(), files.end() );

  for (unsigned ifile=1; ifile<files.size(); ifile++) {
    
    const Observation* obs1 = files[ifile-1]->get_info();
    const Observation* obs2 = files[ifile]->get_info();

    if (! obs1->contiguous(*obs2))
      throw Error (InvalidParam, "dsp::Multifile::load",
		   "'"+files[ifile-1]->get_filename()+"'"
		   " is not contiguous with "
		   "'"+files[ifile]->get_filename()+"'");

  }

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

    uint64 file_bytes = files[index]->get_info()->get_nbytes();

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
