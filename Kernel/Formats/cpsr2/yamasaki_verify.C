/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "yamasaki_verify.h"

using namespace std;

bool yamasaki_verbose = false;

// //////////////////////////////////////////////////////////////////////
//
// YAMASAKI verification
//
#define YAMASAKI_HEADER_SIZE 16
#define YAMASAKI_BLOCK_SIZE 1<<21  // 2 MB

void yamadump (const unsigned char* yama_hdr)
{
  for (int i=0; i<(YAMASAKI_HEADER_SIZE*4); i++) {
    if (i%2 == 0 && isalpha(yama_hdr[i]))
      fprintf (stderr, "%4c ", yama_hdr[i]);
    else
      fprintf (stderr, "0x%2x ", yama_hdr[i]);
    if (i%2)
      fprintf (stderr, ": ");
    if (i%16 == 15)
      fprintf (stderr, "\n");
  }
  fprintf (stderr, "END\n");
}


bool all_zeroes (char* text, int count)
{
  for (int i=0; i<count; i++)
    if (text[i] != 0)
      return false;
  return true;
}

int yama_search (const unsigned char* yama_hdr)
{
  char yamasaki [YAMASAKI_HEADER_SIZE+1];

  char* YAMASAKI = "YAMASAKI";
  int  yama_strlen = strlen (YAMASAKI);

  // search for the YAMASAKI string
  int i=0;
  for (i=0; i<YAMASAKI_HEADER_SIZE; i++)
    yamasaki[i] = yama_hdr[i*2];
  yamasaki[i] = '\0';

  if (strncmp (yamasaki, YAMASAKI, yama_strlen) == 0) {
    cerr << "yama_search - YAMASAKI normal" << endl;
    return 0;
  }
 
  int offset = 0; 
  for (offset = 0; offset < 5; offset++)
    if (strncmp (yamasaki+offset, YAMASAKI, yama_strlen) == 0) {
      cerr << "yama_search - offset by +" << offset << endl;
      return offset*2;
    }
  
  for (offset = 0; offset < 5; offset++)
    if (strncmp (yamasaki, YAMASAKI+offset, yama_strlen-offset) == 0) {
      cerr << "yama_search - offset by -" << offset << endl;
      return -offset*2;
    }

  cerr << "yama_search - no YAMASAKI found - try for 8 zeroes" << endl;
    
  for (offset = 0; offset < 5; offset++)
    if (all_zeroes (yamasaki+offset, yama_strlen)) {
      cerr << "yama_search - offset by +" << offset << endl;
      return offset*2;
    }
  
  for (offset = 0; offset < 5; offset++)
    if (all_zeroes (yamasaki, yama_strlen-offset)) {
      cerr << "yama_search - offset by -" << offset << endl;
      return -offset*2;
    }
  
  cerr << "yama_search - not clearly a FFD2 header" << endl;
  yamadump (yama_hdr);
  return 0;
}

int yamasaki_verify (const char* filename, uint64_t offset_bytes,
		     uint64_t search_offset)
{
  if (search_offset < 2) {
    cerr << "yamasaki_verify - invalid start search_offset " << search_offset
	 << endl;
    return -1;
  }

  int fd = open (filename, O_RDONLY);
  if (fd < 0) {
    fprintf (stderr, "yamasaki_verify - failed open(%s)\n", 
	     filename);
    perror ("yamasaki_verify - ");
    return -1;
  }

  unsigned char yama_hdr [YAMASAKI_HEADER_SIZE*4];

  uint64_t block_size = YAMASAKI_BLOCK_SIZE;
  uint64_t block_count = 0;
  bool testing = true;

  if (lseek (fd, block_size*block_count + search_offset , SEEK_SET) < 0)
    return -1;

  if (read(fd, yama_hdr, YAMASAKI_HEADER_SIZE*4) != YAMASAKI_HEADER_SIZE*4)
    return -1;

  int offset = yama_search (yama_hdr);
  if (offset < 0)
    block_count ++;

  search_offset += offset;

  uint64_t bad_count = 0;
  uint64_t toggle_count = 0;

  bool bad = false;

  while (testing) {
    
    if (lseek (fd, block_size*block_count + search_offset, SEEK_SET) < 0)
      break;

    if (read(fd, yama_hdr, YAMASAKI_HEADER_SIZE) != YAMASAKI_HEADER_SIZE)
      break;
    
    uint64_t expected = offset_bytes +  block_size*block_count;

    uint64_t yama_offset = 0;
    int i=0;

    char yamasaki [YAMASAKI_HEADER_SIZE+1];

    for (i=0; i<YAMASAKI_HEADER_SIZE/2; i++) {
      yamasaki[i] = yama_hdr[2*i];
      yama_offset += uint64_t(yama_hdr[2*i+1]) << (8*i);
    }
    yamasaki[i] = '\0';
    yama_offset *= 2;	/* convert from word to byte counter */


    if (strncmp(yamasaki, "YAMASAKI", 8))  {
      if (yamasaki_verbose) {
	fprintf (stderr, "yamasaki_verify: incorrect header for block "
		 UI64"\n", block_count);
	yamadump (yama_hdr);
      }
    }

    if (yama_offset != expected) {
      if (yamasaki_verbose) {
	fprintf (stderr, "yamasaki_verify: Byte count mismatch for block "
		 UI64" value "UI64" expected "UI64"\n",
		 block_count, yama_offset, expected);
	yamadump (yama_hdr);
      }
      bad_count ++;
      bad = true;
    }
    else  {
      if (bad)
	toggle_count ++;
      bad = false;
    }
    block_count++;
  }

  close (fd);
  
  if (bad_count) {
    fprintf (stdout, "yamasaki_verify: "UI64" bad blocks out of "UI64"\n",
	     bad_count, block_count);
    
    if (toggle_count) {
      fprintf (stdout, "yamasaki_verify: data came good again "UI64
	       " out of "UI64" times\n", toggle_count, bad_count);
      if (toggle_count >= bad_count -1)
	return 0;
    }

    return -1;
  }

  fprintf (stderr, "yamasaki_verify: "UI64" counts verified\n",block_count);
  return 0;
}
