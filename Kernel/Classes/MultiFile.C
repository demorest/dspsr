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
    old_filenames[i] = files[i].filename;

  // If there is no loader, create one from the first file
  if (!loader)
    loader = File::create( new_filenames[0],bs_index );

  // open up each of the new files and add it to our list of files
  for( unsigned i=0; i<new_filenames.size(); i++){
    if( !is_one_of(new_filenames[i],old_filenames) ){

      loader->open( new_filenames[i],bs_index );
      files.push_back( PseudoFile( loader ) );

      if( verbose )
	cerr << "dsp::MultiFile::open new PseudoFile = " 
	     << files.back().filename << endl;

    }
  }

  ensure_contiguity();
  setup();
}

void dsp::MultiFile::open(const vector<PseudoFile*>& pseudos){
  if (pseudos.empty())
    throw Error (InvalidParam, "dsp::Multifile::open",
		 "An empty list of PseudoFiles has been given to this method");

  // construct a list of the files we already have open
  vector<string> old_filenames (files.size());
  for (unsigned i=0; i<files.size(); i++)
    old_filenames[i] = files[i].filename;

  // If there is no loader, create one from the first pseudo
  if (!loader){
    loader = File::create( pseudos.front()->filename, pseudos.front()->bs_index );
    loader->open( *pseudos.front() );
  }

  // Add each element of 'pseudos' to our list of PseudoFiles
  for( unsigned i=0; i<pseudos.size(); i++){
    if( !pseudos[i] )
      throw Error(InvalidState,"dsp::MultiFile::open()",
		  "pseudos[%d] was NULL",i);
    if( is_one_of(pseudos[i]->filename,old_filenames) )
      continue;

    files.push_back( *pseudos[i] );

    if( verbose )
      cerr << "dsp::MultiFile::open new PseudoFile = " 
	   << files.back().filename << endl;
  } 

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

  loader->open (files.front());

  current_index = 0;
  current_filename = files.front().filename;

  reset();
}

//! Makes sure only these filenames are open
void dsp::MultiFile::have_open (const vector<string>& filenames,int bs_index)
{
  // Erase any files we already have open that we don't want open
  for( unsigned ifile=0; ifile<files.size(); ifile++){
    if( !is_one_of(files[ifile].filename,filenames) ){
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

    Observation* obs1 = &files[ifile-1];
    Observation* obs2 = &files[ifile];

    if( verbose )
      fprintf(stderr,"dsp::MultiFile::ensure_contiguity() Going to call contiguous() with obs1.start=%s obs1.end=%s obs2.start=%s obs2.end=%s\n",
	      obs1->get_start_time().printall(),
	      obs1->get_end_time().printall(),
	      obs2->get_start_time().printall(),
	      obs2->get_end_time().printall());

    if ( !obs1->contiguous(*obs2) ){
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
    throw Error(InvalidState, "dsp::MultiFile::seek_bytes",
		"no loader.  Have you called MultiFile::open() yet?");

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

bool dsp::MultiFile::has_loader ()
{
  return loader;
}

dsp::File* dsp::MultiFile::get_loader ()
{
  return loader;
}

// HSK 24 June 2003 This function is a disaster- what it's original intention is unclear to me.  What about loader->get_next_sample()- wouldn't that return the same thing?
uint64 dsp::MultiFile::get_next_sample(){
  uint64 samples_over = 0;

  for( unsigned i=0; i<current_index; i++)
    samples_over += files[i].get_ndat();

  if( verbose )
    fprintf(stderr,"dsp::MultiFile::get_next_sample() got get_load_sample()="UI64" and samples_over="UI64" and current_index=%d\n",
	    get_load_sample(), samples_over,current_index);

  //return get_load_sample();
  return get_load_sample()-samples_over;
}

