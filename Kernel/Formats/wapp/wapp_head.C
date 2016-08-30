#include "wapp_head.h"
#include "machine_endian.h"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include "Error.h"

using namespace std;

void readheader(int fd, WAPP_HEADER* hdr)
{
  char c;
  int bytes_read;
  bool skip=1;
  while(skip)
  {
    if((bytes_read=read(fd,(char *)(&c),sizeof(c)))==-1)
      throw Error(FailedSys,"readheader","Unable to read from input file");
    if(c==(char)(0))
      skip=0;
  }
  bytes_read=read(fd,(char *)(&(hdr->header_version)),sizeof(hdr->header_version));
  FromLittleEndian(hdr->header_version);
  bytes_read=read(fd,(char *)(&(hdr->header_size)),sizeof(hdr->header_size));
  FromLittleEndian(hdr->header_size);
  bytes_read=read(fd,(char *)(&(hdr->obs_type)),hdr->header_size-sizeof(hdr->header_size)-sizeof(hdr->header_version));
  FromLittleEndian(hdr->obs_time);
  FromLittleEndian(hdr->samp_time);
  FromLittleEndian(hdr->wapp_time);
  FromLittleEndian(hdr->num_lags);
  FromLittleEndian(hdr->nifs);
  FromLittleEndian(hdr->level);
  FromLittleEndian(hdr->lagformat);
  FromLittleEndian(hdr->lagtrunc);
  FromLittleEndian(hdr->cent_freq);
  FromLittleEndian(hdr->bandwidth);
  FromLittleEndian(hdr->freqinversion);
  FromLittleEndian(hdr->src_ra);
  FromLittleEndian(hdr->src_dec);
  FromLittleEndian(hdr->start_az);
  FromLittleEndian(hdr->start_za);
  FromLittleEndian(hdr->start_ast);
  FromLittleEndian(hdr->start_lst);
  FromLittleEndian(hdr->sum);
  FromLittleEndian(hdr->psr_dm);
  FromLittleEndian(hdr->dumptime);
  FromLittleEndian(hdr->nbins);
  if(hdr->header_version!=7)
    throw Error(InvalidState,"readheader","Header version %d do not match 7!",
                hdr->header_version);
  if(hdr->header_size!=2440)
    throw Error(InvalidState,"readheader","Header size incorrect!");
}

