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

void dsp::MultiFile::open (const vector<string>& new_filenames)
{
  if (new_filenames.empty())
    throw Error (InvalidParam, "dsp::Multifile::open",
		 "An empty list of filenames has been given to this method");

  // construct a list of the files we already have open
  vector<string> old_filenames (files.size());
  for (unsigned i=0; i<files.size(); i++)
    old_filenames[i] = files[i].filename;

  // If there is no loader, create one from the first file
  if (!loader)
    loader = File::create( new_filenames[0] );

  // open up each of the new files and add it to our list of files
  for( unsigned i=0; i<new_filenames.size(); i++){
    if( !is_one_of(new_filenames[i],old_filenames) ){

      loader->open( new_filenames[i] );
      files.push_back( PseudoFile( loader ) );

      if( verbose )
	cerr << "dsp::MultiFile::open new PseudoFile = " 
	     << files.back().filename << endl;

    }
  }

  loader->open( new_filenames.front() );

  ensure_contiguity();

  setup();
}

void dsp::MultiFile::setup ()
{
  info = files.front();

  uint64 total_ndat = 0;
  for( unsigned i=0; i<files.size(); i++)
    total_ndat += files[i].get_ndat();

  info.set_ndat (total_ndat);

  // MultiFile must reflect the time sample resolution of the underlying device
  resolution = loader->resolution;

  set_loader (0);

  reset();
}

//! Makes sure only these filenames are open
void dsp::MultiFile::have_open (const vector<string>& filenames)
{
  // Erase any files we already have open that we don't want open
  for( unsigned ifile=0; ifile<files.size(); ifile++){
    if( !is_one_of(files[ifile].filename,filenames) ){
      files.erase(files.begin()+ifile);
      ifile--;
    }
  }

  open (filenames);
}

//! Erase the entire list of loadable files
void dsp::MultiFile::erase_files()
{
  files.erase( files.begin(), files.end());
  delete loader.release();
  info = Observation();
  reset();
}

//! Erase just some of the list of loadable files
void dsp::MultiFile::erase_files(const vector<string>& erase_filenames)
{
  for( unsigned ifile=0; ifile<files.size(); ifile++){
    if( is_one_of(files[ifile].filename,erase_filenames) ){
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

void dsp::MultiFile::ensure_contiguity()
{
  if( verbose )
    cerr << "dsp::MultiFile::ensure_contiguity enter" << endl;

  sort( files.begin(), files.end() );

  for (unsigned ifile=1; ifile<files.size(); ifile++) {
    if( verbose )
      cerr << "dsp::MultiFile::ensure_contiguity files " << ifile-1 
	   << " and " << ifile << endl;

    const Observation* obs1 = &files[ifile-1];
    const Observation* obs2 = &files[ifile];

    if (! obs1->contiguous(*obs2)){
      char cstr[4096];
      sprintf(cstr,"file %d (%s) is not contiguous with file %d (%s)",
	      ifile-1,files[ifile-1].filename.c_str(),
	      ifile,files[ifile].filename.c_str());
      throw Error (InvalidParam, "dsp::Multifile::ensure_contiguity",cstr);
    }

  }

  if( verbose )
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
    throw Error(InvalidState,"dsp::MultiFile::seek_bytes",
		"Seek bytes called with no loader.  Have you called MultiFile::open() yet?");

  if (verbose)
    cerr << "MultiFile::seek_bytes nbytes=" << bytes << endl;

  // Total number of bytes stored in files thus far
  uint64 total_bytes = 0;

  unsigned index;
  for (index = 0; index < files.size(); index++) {

    // Number of bytes stored in this file
    uint64 file_bytes = files[index].get_nbytes();

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

  loader->open (files[index]);

  current_index = index;
  current_filename = files[index].filename;
}
