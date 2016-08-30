/***************************************************************************
 *
 *   Copyright (C) 2011, 2013 by James M Anderson  (MPIfR)
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#if HAVE_CONFIG_H
#include <config.h>
#endif
#include "dsp/LuMPFile.h"
#include "dsp/LuMPObservation.h"
#include "ascii_header.h"
#include "Error.h"

#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

using namespace std;

dsp::LuMPFile::LuMPFile (const char* filename) : File ("LuMP"),
                                                 file_data_position(-1)
{
    if (verbose)
        cerr << "dsp::LuMPFile::LuMPFile used" << endl;
    if (filename) {
        open (filename);
    }
}

string dsp::LuMPFile::get_header (const char* filename)
{
  std::string line;
  std::ifstream input (filename);

  if (!input)
    return line;

  std::getline (input, line, '\0');
  //if (verbose)
  //    std::cerr << "dsp::LuMPFile::get_header header " << line << endl;

  return line;
}

bool dsp::LuMPFile::is_valid (const char* filename) const
{
  string header = get_header (filename);

  if (header.empty())
  {
    if (verbose)
      cerr << "dsp::LuMPFile::is_valid empty header" << endl;
    return false;
  }

  // verify that the buffer read contains a valid LuMP header
  float version;
  if (ascii_header_get (header.c_str(), "LUMP_VERSION", "%f", &version) < 0)
  {
    if (verbose)
      cerr << "dsp::LuMPFile::is_valid LUMP_VERSION not defined" << endl;
    return false;
  }

  return true;

  LuMPObservation data (filename);
  return true;
}

void dsp::LuMPFile::open_file (const char* filename)
{
  string header = dsp::LuMPFile::get_header (filename);

  if (header.empty())
    throw Error (FailedCall, "dsp::LuMPFile::open_file",
		 "get_header(%s) failed", filename);
  
  info = new LuMPObservation (header.c_str());

  unsigned hdr_size;
  if (ascii_header_get (header.c_str(), "HDR_SIZE", "%u", &hdr_size) < 0)
    throw Error (FailedCall, "dsp::LuMPFile::open_file",
		 "ascii_header_get(HDR_SIZE) failed");
  if (verbose)
      cerr << "header size = " << hdr_size << " bytes" << endl;
  header_bytes = hdr_size;

  if(get_lump_info()->get_read_from_LuMP_file())
  {
    // open the file
    fd = ::open (filename, O_RDONLY);
    if (fd < 0)
      throw Error (FailedSys, "dsp::LuMPFile::open_file()", 
                   "open(%s) failed", filename);
    file_data_position = -header_bytes;    
  }
  else
  {
    // use STDIN
    fd = STDIN_FILENO;
    file_data_position = 0;  // STDIN pipe feeds only data.
  }
  
  // cannot load less than a byte. set the time sample resolution accordingly
  unsigned bits_per_byte = 8;
  resolution = bits_per_byte / get_info()->get_nbit();
  if (resolution == 0)
    resolution = 1;

  if (verbose)
    cerr << "LuMPFile::open exit" << endl;
}


//! Load bytes from file
int64_t dsp::LuMPFile::load_bytes (unsigned char* buffer, uint64_t bytes)
{
    // if (verbose)
    //     cerr << "dsp::LuMPFile::load_bytes nbytes=" << bytes << endl;

  ssize_t bytes_read(0);
  size_t bytes_request(bytes);
  uint64_t total_bytes_read(0);
  int64_t old_pos(0);
  int64_t new_pos(0);
  int64_t end_pos(0);
  bool keep_reading(true);
  end_of_data = false;

  if(get_lump_info()->get_read_from_LuMP_file())
  {
    // Based on File.C code, with extra checking to allow for read
    // returning less data than requested (allowed by POSIX standard)
    old_pos = lseek(fd,0,SEEK_CUR);

    do {
      bytes_read = read (fd, buffer+total_bytes_read, bytes_request);

      if(bytes_read == ssize_t(bytes_request))
      {
        total_bytes_read += bytes_read;
        bytes_request -= bytes_read;
        bytes_read = total_bytes_read;
        keep_reading = false;
      }
      else if(bytes_read == 0)
      {
        // End of file
        bytes_read = total_bytes_read;
        keep_reading = false;
      }
      else if(bytes_read > 0)
      {
        // Only partially read what was requested.  Keep trying
        total_bytes_read += bytes_read;
        bytes_request -= bytes_read;
      }
      else
      {
        // Some sort of failure
        int errno_save = errno; // The value of errno is not guaranteed
                                // to survive a library call.
        perror ("dsp::LuMPFile::load_bytes read error");
        if( (errno_save == EAGAIN)
          || (errno_save == EWOULDBLOCK)
          || (errno_save == EINTR) )
        {
          // Try again
        }
        else {
          // I/O failure
          end_of_data = true;
          keep_reading = false;
        }
      }
    } while(keep_reading);

    new_pos = lseek(fd,0,SEEK_CUR);
    file_data_position = new_pos - int64_t(header_bytes);
    end_pos = get_info()->get_nbytes() + uint64_t(header_bytes);
  }
  else
  {
    // Read from STDIN.  Don't seek.
    if(file_data_position < 0)
    {
        cerr << "dsp::LuMPFile::load_bytes Warning: request to read data after file rewind.  Garbage results will likely be produced." << endl;
        file_data_position = 0;
    }
    old_pos = file_data_position + int64_t(header_bytes);

    do {
      bytes_read = read (fd, buffer+total_bytes_read, bytes_request);

      if(bytes_read == ssize_t(bytes_request))
      {
        total_bytes_read += bytes_read;
        bytes_request -= bytes_read;
        bytes_read = total_bytes_read;
        keep_reading = false;
      }
      else if(bytes_read == 0)
      {
        // End of file
        bytes_read = total_bytes_read;
        keep_reading = false;
      }
      else if(bytes_read > 0)
      {
        // Only partially read what was requested.  Keep trying
        total_bytes_read += bytes_read;
        bytes_request -= bytes_read;
      }
      else
      {
        // Some sort of failure
        int errno_save = errno; // The value of errno is not guaranteed
                                // to survive a library call.
        if( (errno_save == EAGAIN)
          || (errno_save == EWOULDBLOCK)
          || (errno_save == EINTR) )
        {
          // Try again
            
          // Do not generate a warning here.  As STDIN may be non-blocking
          // on some systems, or non-blocking from the program sending us
          // data, the "try again" codes may be common.  Just give the sending
          // program some CPU time to generate more data.
          int yield_return = ::sched_yield();
          if((yield_return))
          {
            perror ("dsp::LuMPFile::load_bytes sched_yield error while waiting for data sending program to send more data");
          }
        }
        else {
          // I/O failure
          perror ("dsp::LuMPFile::load_bytes read error");
          end_of_data = true;
          keep_reading = false;
        }
      }
    } while(keep_reading);

    file_data_position += int64_t(total_bytes_read);

    new_pos = file_data_position + int64_t(header_bytes);
    end_pos = get_info()->get_nbytes() + uint64_t(header_bytes);
  }

  // if (verbose)
  //   cerr << "dsp::LuMPFile::load_bytes bytes_read=" << bytes_read
  //        << " old_pos=" << old_pos << " new_pos=" << new_pos 
  //        << " end_pos=" << end_pos << endl;

  if(bytes_read > 0)
  {
    if( uint64_t(new_pos) >= end_pos ){
      bytes_read = ssize_t(end_pos - old_pos);
      if(get_lump_info()->get_read_from_LuMP_file())
      {
        lseek(fd,end_pos,SEEK_SET);
      }
      end_of_data = true;
    }
  }

  return bytes_read;
}

//! Adjust the file pointer
int64_t dsp::LuMPFile::seek_bytes (uint64_t bytes)
{
  // if (verbose)
  //   cerr << "dsp::LuMPFile::seek_bytes nbytes=" << bytes 
  //        << " header_bytes=" << header_bytes << endl;
  
  if (fd < 0)
    throw Error (InvalidState, "dsp::LuMPFile::seek_bytes", "invalid fd");


  if(get_lump_info()->get_read_from_LuMP_file())
  {
    // Data reading for file copied directly from File.C code
    bytes += header_bytes;
    int64_t retval = lseek (fd, bytes, SEEK_SET);
    if (retval < 0)
      throw Error (FailedSys, "dsp::LuMPFile::seek_bytes", "lseek error");
      
    if( uint64_t(retval) == get_info()->get_nbytes()+uint64_t(header_bytes) )
      end_of_data = true;
    else
      end_of_data = false;

    // return absolute data byte offset from the start of file
    return retval - header_bytes;
  }
  else
  {
    // Read from STDIN.  Don't seek.

    // Are we asked to go to the current location?
    if(int64_t(bytes) == file_data_position)
    {
    }
    else if(bytes == 0)
    {
      // Instruction to rewind data file.  As long as we don't read
      // anything later on, this should be fine.  Note that file_data_position
      // is set to 0 on initialization, so a seek to 0 bytes offset in the
      // data at the start of data-reading is captured by the if statement
      // above.
      file_data_position = -1;
    }
    else if(int64_t(bytes) > file_data_position)
    {
      if(file_data_position == -1)
      {
        cerr << "dsp::LuMPFile::seek_bytes Warning: request to seek in data after file rewind.  Garbage results will likely be produced." << endl;
        file_data_position = 0;
      }
      // Skip forward in datastream.  This *can* be done on a pipe.
      const uint64_t BUF_SIZE(16384);
      unsigned char mybuffer[BUF_SIZE];
      
      while(file_data_position < int64_t(bytes))
      {
        uint64_t bytes_to_read = bytes - uint64_t(file_data_position);
        if(bytes_to_read > BUF_SIZE)
          bytes_to_read = BUF_SIZE;

        int64_t bytes_read = load_bytes(mybuffer, bytes_to_read);

        if (bytes_read < 0)
        {
          file_data_position = -1;
          throw Error (FailedSys, "dsp::LuMPFile::seek_bytes", "read error for pipe forward skip");
        }
        else if (bytes_read < int64_t(bytes_to_read))
        {
          file_data_position = -1;
          end_of_data = true;
          break;
        }

        file_data_position += int64_t(bytes_read);
      }
    }
    else
    {
      // Skip backward in datastream.  This *cannot* be done on a pipe.
      cerr << "dsp::LuMPFile::seek_bytes Error: request to seek backward in data is not possible on a pipe." << endl;
      file_data_position = -1;
      end_of_data = true;
      throw Error (FailedSys, "dsp::LuMPFile::seek_bytes", "lseek error");
    }
  }

    
  if(file_data_position == int64_t(get_info()->get_nbytes()))
    end_of_data = true;
      
  if(file_data_position >= 0)
    return file_data_position;
  return 0;
}

//! Return ndat given the file and header sizes, nchan, npol, and ndim
/*! Called by open_file for some file types, to determine that the
header ndat matches the file size.  Requires 'info' parameters
nchan, npol, and ndim as well as header_bytes to be correctly set */
int64_t dsp::LuMPFile::fstat_file_ndat(uint64_t tailer_bytes)
{ 
  if(get_lump_info()->get_read_from_LuMP_file())
  {
    return dsp::File::fstat_file_ndat(tailer_bytes);
  }

  // If we reach this point, then we are reading from STDIN, and cannot
  // fstat the file to see how much data is present.  Use the info data.

  uint64_t data_bytes = get_lump_info()->get_LuMP_file_size()
                        - header_bytes - tailer_bytes;
  if(get_lump_info()->get_LuMP_file_size() == 0)
  {
    data_bytes = 0;
  }
  
  // if( verbose )
  //   cerr << "dsp::LuMPFile::fstat_file_ndat(): buf=" << get_lump_info()->get_LuMP_file_size()
  //        << " header_bytes=" << header_bytes 
  //        << " tailer_bytes=" << tailer_bytes
  //        << " data_bytes=" << data_bytes << endl;

  return get_info()->get_nsamples (data_bytes);
}

