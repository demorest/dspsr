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

// //! operator used to sort the File vector
//bool operator < (const Reference::To<dsp::File>& f1,
//	 const Reference::To<dsp::File>& f2)
//{
//return f1->get_info()->get_start_time() < f2->get_info()->get_start_time();
//}


// This is not the best way to do the "operator <" - it should be done in the PseudoFile class
// This method also exists in the PseudoFile Class so that proper compilers (not gcc version 2) can access it properly
bool operator < (const dsp::PseudoFile& f1, const dsp::PseudoFile& f2){
  return f1.get_start_time() < f2.get_start_time();
}

void dsp::MultiFile::open (vector<string> new_filenames)
{
  if (new_filenames.empty())
    throw Error (InvalidParam, "dsp::Multifile::open()", "An empty list of filenames has been given to this method");

  // construct a list of the files we already have open
  vector<string> old_filenames(files.size());
  for( unsigned i=0; i<files.size(); i++)
    old_filenames[i] = files[i].filename;

  // create the opener
  Reference::To<File> opener;
  if( !files.empty() )
    opener = loader->clone();
  else
    opener = File::create( new_filenames[0] );

  // open up each of the new files and add it to our list of files
  for( unsigned i=0; i<new_filenames.size(); i++){
    if( !is_one_of(new_filenames[i],old_filenames) ){
      opener->open( new_filenames[i] );

      files.push_back(PseudoFile());
      files.back().filename = new_filenames[i];
      if( verbose )
	fprintf(stderr,"dsp::MultiFile::open() has pushed back a file of '%s'\n",files.back().filename.c_str());
      files.back().header_bytes = opener->get_header_bytes();
      files.back().Observation::operator=( *opener->get_info() );
    }
  }

  ensure_contiguity();

  setup(opener);
}

void dsp::MultiFile::setup(Reference::To<dsp::File> opener){
  info = files.front();

  uint64 total_ndat = 0;
  for( unsigned i=0; i<files.size(); i++)
    total_ndat += files[i].get_ndat();

  info.set_ndat (total_ndat);

  if( !loader )
    loader = opener;
  loader->open( files.front().filename, &files.front() );

  // MultiFile must reflect the time sample resolution of the underlying device
  resolution = loader->resolution;

  current_filename = files.front().filename;

  reset();
}

//! Makes sure only these filenames are open
void dsp::MultiFile::have_open(vector<string> filenames){

  if( files.empty() ){
    open(filenames);
    return;
  }

  // Erase any files we already have open that we don't want open
  for( unsigned ifile=0; ifile<files.size(); ifile++){
    if( !is_one_of(files[ifile].filename,filenames) ){
      files.erase(files.begin()+ifile);
      ifile--;
    }
  }

  // Make a list of files we still have open
  vector<string> old_filenames; 
  for( unsigned ifile=0; ifile<files.size(); ifile++)
    old_filenames.push_back( files[ifile].filename );

  Reference::To<File> opener(loader->clone());

  // Open any files we don't already have open
  for( unsigned i=0; i<filenames.size(); i++){
    if( !is_one_of(filenames[i],old_filenames) ){
      opener->open( filenames[i] );

      files.push_back( PseudoFile() );
      files.back().filename = filenames[i];
      files.back().header_bytes = opener->get_header_bytes();
      files.back().Observation::operator=( *opener->get_info() );
    }
  }

  // sort and ensure contiguity
  ensure_contiguity();

  setup(loader);
}

//! Erase the entire list of loadable files
void dsp::MultiFile::erase_files(){
  files.erase( files.begin(), files.end());
  delete loader.release();
  info = Observation();
  reset();
}

//! Erase just some of the list of loadable files
void dsp::MultiFile::erase_files(vector<string> erase_filenames){
  for( unsigned ifile=0; ifile<files.size(); ifile++){
    if( is_one_of(files[ifile].filename,erase_filenames) ){
      files.erase( files.begin()+ifile );
      ifile--;
    }
  }
  
  if( files.empty() ){
    info = Observation();
    reset();
    delete loader.release();
    return;
  }

  ensure_contiguity();

  setup(loader);
}

void dsp::MultiFile::ensure_contiguity(){
  if( verbose )
    fprintf(stderr,"In dsp::MultiFile::ensure_contiguity()\n");

  sort( files.begin(), files.end() );

  for (unsigned ifile=1; ifile<files.size(); ifile++) {
    if( verbose )
      fprintf(stderr,"Ensuring contiguity for files %d and %d\n",ifile-1,ifile);

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
    fprintf(stderr,"Returning from dsp::MultiFile::ensure_contiguity()\n");
}

//! Load bytes from file
int64 dsp::MultiFile::load_bytes (unsigned char* buffer, uint64 bytes)
{
  if (verbose)
    cerr << "MultiFile::load_bytes nbytes=" << bytes << endl;
  
  if( !loader )
    throw Error(InvalidState,"dsp::MultiFile::load_bytes()",
		"Load bytes called with no loader.  Have you called MultiFile::open() yet?");

  uint64 bytes_loaded = 0;

  while (bytes_loaded < bytes) {

    int64 to_load = bytes - bytes_loaded;

    if (index >= files.size()) {
      if (verbose)
	cerr << "MultiFile::load_bytes end of data" << endl;
      end_of_data = true;
      break;
    }

    // Ensure we are loading from correct file
    if( loader->get_current_filename() != files[index].filename ){
      loader->open( files[index].filename, &files[index] );
      current_filename = files[index].filename;
    }

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
    throw Error(InvalidState,"dsp::MultiFile::seek_bytes()",
		"Seek bytes called with no loader.  Have you called MultiFile::open() yet?");

  if (verbose)
    cerr << "MultiFile::seek_bytes nbytes=" << bytes << endl;

  // Total number of bytes stored in files thus far
  uint64 total_bytes = 0;

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

  if( loader->get_current_filename() != files[index].filename ){
    loader->open( files[index].filename, &files[index] );
    current_filename = files[index].filename;
  }

  int64 seeked = loader->seek_bytes (bytes-total_bytes);
  if (seeked < 0)
    return -1;

  return total_bytes + seeked;
}
