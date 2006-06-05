#include "dsp/MultiFile.h"
#include "dsp/File.h"
#include "Error.h"
#include "genutil.h"

#include <algorithm>
#include <math.h>

dsp::MultiFile::MultiFile () : Seekable ("MultiFile")
{
  current_index = 0;
}

dsp::MultiFile::~MultiFile ()
{
}

/*! This method adds to the current set of input files and re-sorts
  them all files by start time.
  
  \post Resets the file pointers 
*/
void dsp::MultiFile::open (const vector<string>& new_filenames, int bs_index)
{
  if (new_filenames.empty())
    throw Error (InvalidParam, "dsp::Multifile::open",
		 "An empty list of filenames has been given to this method");

  // construct a list of the files we already have open
  vector<string> old_filenames (files.size());
  for (unsigned i=0; i<files.size(); i++)
    old_filenames[i] = files[i]->get_filename();

  // open up each of the new files and add it to our list of files
  for( unsigned i=0; i<new_filenames.size(); i++){
    if( !is_one_of(new_filenames[i],old_filenames) ){

      // If there is no loader, create one from the first file
      loader = File::create( new_filenames[i], bs_index );

      files.push_back( loader );

      loader->close();

      if (verbose)
	cerr << "dsp::MultiFile::open new PseudoFile = " 
	     << files.back()->get_filename() << endl;
    }
  }

  ensure_contiguity();
  setup();
}

void dsp::MultiFile::setup ()
{
  info = *(files[0]->get_info());

  uint64 total_ndat = 0;
  for( unsigned i=0; i<files.size(); i++)
    total_ndat += files[i]->get_info()->get_ndat();

  info.set_ndat (total_ndat);

  // MultiFile must reflect the time sample resolution of the underlying device
  resolution = loader->resolution;

  loader = files[0];
  loader->reopen();

  current_index = 0;
  current_filename = files[0]->get_filename();

  reset();
}

//! Makes sure only these filenames are open
void dsp::MultiFile::have_open (const vector<string>& filenames,int bs_index)
{
  // Erase any files we already have open that we don't want open
  for( unsigned ifile=0; ifile<files.size(); ifile++){
    if( !is_one_of(files[ifile]->get_filename(),filenames) ){
      files.erase(files.begin()+ifile);
      ifile--;
    }
  }

  open (filenames,bs_index);
}

//! Erase the entire list of loadable files
void dsp::MultiFile::erase_files()
{
  files.erase( files.begin(), files.end());
  loader = 0;
  info = Observation();
  reset();
}

//! Erase just some of the list of loadable files
void dsp::MultiFile::erase_files(const vector<string>& erase_filenames)
{
  for( unsigned ifile=0; ifile<files.size(); ifile++){
    if( is_one_of(files[ifile]->get_filename(),erase_filenames) ){
      files.erase( files.begin()+ifile );
      ifile--;
    }
  }
  
  if( files.empty() ){
    erase_files ();
    return;
  }

  ensure_contiguity();
  setup();
}

bool time_order (const dsp::File* a, const dsp::File* b)
{
  return a->get_info()->get_start_time() < b->get_info()->get_start_time();
}

void dsp::MultiFile::ensure_contiguity()
{
  if (verbose)
    cerr << "dsp::MultiFile::ensure_contiguity enter" << endl;

  sort( files.begin(), files.end(), time_order );

  for (unsigned ifile=1; ifile<files.size(); ifile++) {
    if (verbose)
      cerr << "dsp::MultiFile::ensure_contiguity files " << ifile-1 
	   << " and " << ifile << endl;

    Observation* obs1 = files[ifile-1]->get_info();
    Observation* obs2 = files[ifile]->get_info();;

    if (verbose)
      cerr << "dsp::MultiFile::ensure_contiguity"
	" obs.start=" << obs1->get_start_time() << 
	" obs1.end=" << obs1->get_end_time() << 
	" obs2.start=" << obs2->get_start_time() << 
	" obs2.end=" << obs2->get_end_time() << endl;

    if ( !obs1->contiguous(*obs2) ){
      char cstr[4096];
      sprintf(cstr,"file %u (%s) is not contiguous with file %u (%s)",
	      ifile-1,files[ifile-1]->get_filename().c_str(),
	      ifile,files[ifile]->get_filename().c_str());
      throw Error (InvalidParam, "dsp::Multifile::ensure_contiguity",cstr);
    }

  }

  if (verbose)
    cerr << "dsp::MultiFile::ensure_contiguity return" << endl;
}

//! Load bytes from file
int64 dsp::MultiFile::load_bytes (unsigned char* buffer, uint64 bytes)
{
  if (verbose)
    cerr << "MultiFile::load_bytes nbytes=" << bytes << endl;
  
  if( !loader )
    throw Error(InvalidState,"dsp::MultiFile::load_bytes",
		"No loader.  Possible MultiFile::open failure.");

  uint64 bytes_loaded = 0;
  unsigned index = current_index;

  while (bytes_loaded < bytes) {

    int64 to_load = bytes - bytes_loaded;

    if (index >= files.size()) {
      if (verbose)
	cerr << "dsp::MultiFile::load_bytes end of data" << endl;
      end_of_data = true;
      break;
    }

    // Ensure we are loading from correct file
    set_loader (index);

    int64 did_load = loader->load_bytes (buffer, to_load);

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
  if( !loader )
    throw Error(InvalidState, "dsp::MultiFile::seek_bytes",
		"no loader.  Have you called MultiFile::open() yet?");

  if (verbose)
    cerr << "MultiFile::seek_bytes nbytes=" << bytes << endl;

  // Total number of bytes stored in files thus far
  uint64 total_bytes = 0;

  unsigned index;
  for (index = 0; index < files.size(); index++) {

    // Number of bytes stored in this file
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

  set_loader (index);

  int64 seeked = loader->seek_bytes (bytes-total_bytes);
  if (seeked < 0)
    return -1;

  return total_bytes + seeked;
}

void dsp::MultiFile::set_loader (unsigned index)
{
  if (index == current_index)
    return;

  loader = files[index];

  // MiniFile requires loader to know what output->get_rawptr() is
  // (ASSUMPTION: output is what is getting loaded into rather than some other BitSeries via load())
  loader->set_output( get_output() );
  loader->reopen();

  current_index = index;
  current_filename = files[index]->get_filename();
}

bool dsp::MultiFile::has_loader ()
{
  return loader;
}

dsp::File* dsp::MultiFile::get_loader ()
{
  return loader;
}
