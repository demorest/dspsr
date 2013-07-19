/***************************************************************************
 *
 *   Copyright (C) 2004 by Craig West
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/Mark4File.h"
#include <iomanip>

#include "Error.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

using namespace std;

dsp::Mark4File::Mark4File (const char* filename) 
  : File ("Mark4")
{
  
  channels = 8; 
  SYNC_pattern = 0xFF;
}

dsp::Mark4File::~Mark4File ()
{

}

bool dsp::Mark4File::is_valid (const char* filename) const
{
  int filed = ::open(filename, O_RDONLY);
  
  if(filed == -1)
    throw Error (FailedSys, "dsp::Mark4File::open",
		 "failed fopen(%s)", filename);
  
  int chans = count_channels(filed);
  if (chans == 0)
    return false;
  else
    return true;
  
}

void dsp::Mark4File::open_file (const char* filename)
{
  
  fd = ::open(filename, O_RDONLY);
  
  if(fd == -1)
    throw Error (FailedSys, "dsp::Mark4File::open",
		 "failed fopen(%s)", filename);
  
  //  cerr << endl << "Counting channels ..." << endl;
  channels = count_channels (fd);
  //  cerr << "Located " << channels << " channels" << endl << endl;
  
  //  cout << "Channels: " << channels << endl;

  // testing
  //  cerr << "LOCATING SYNC..." << endl;
  uint64_t first_sync = find_sync(fd);
  //  cerr << "NextSYNC: " << first_sync << "\t" << hex << first_sync << dec << endl;

  //  lseek(fd,first_sync+(4*channels),SEEK_CUR);
  
  //  cerr << "LOCATE SECOND SYNC..." << endl;
  uint64_t next_sync = find_sync(fd,first_sync+(4*channels));
  //  cerr << "NextSYNC: " << next_sync << "\t" << hex << next_sync << dec << endl;
    
  //  cerr << "Distances between SYNCs: " << next_sync - first_sync << endl;

  uint64_t sync_distance = next_sync-first_sync;
  
  if(sync_distance == 2520*channels){ // Mark4 VLBA mode;
    mode = VLBA;
    if(verbose)
      cerr << "Running in VLBA mode" << endl;
  }
  else if(sync_distance == 2500*channels){ // Standard Mark4 mode
    mode = Standard;
    if(verbose)
      cerr << "Running in Standard mode" << endl;
  }
  else
    cerr << "Unknown mode - distance between SYNCS: " << sync_distance << endl;
  
  //  cerr << endl << endl;
  
  get_info()->set_start_time(decode_date());
  
  //  cout << "Decoded date: " << get_info()->get_start_time().printall() << endl;
  
  // Need to rewind time to start of file, which is NOT the start of the header.
  
  
  // These need to be auto set.
  get_info()->set_bandwidth(16.0);
  get_info()->set_rate(32000000);
  get_info()->set_state(Signal::Nyquist);

  get_info()->set_machine("Mark4");
  
  get_info()->set_npol(2);
  get_info()->set_nbit(2);

  get_info()->set_nchan(1);
  
  //ASSUMES THE DATA IS 16MHz channels
  //  get_info()->set_nchan(int(channels/16));
  unsigned bits_per_byte = 8;
  resolution = bits_per_byte / get_info()->get_nbit();
  
  struct stat file_info;
  
  stat (filename, &file_info);
  
  // This needs to include information about the headers in Mark4-VLBA modes
  get_info()->set_ndat( int64_t((file_info.st_size)/get_info()->get_npol() )*16/(get_info()->get_nbit()*get_info()->get_npol()*get_info()->get_nchan()));
  
  uint64_t last_sync = find_sync(fd,file_info.st_size-2500*channels);
  //  cout << last_sync << "\t" << decode_date(last_sync).printall() << endl;
  
  double initial_offset_time = ((first_sync-8*channels)/(get_info()->get_npol()))
                                *(8/get_info()->get_nbit())/get_info()->get_rate();

  //  cout << get_info()->get_start_time().printall() << "\t" << first_sync << "\t" << initial_offset_time << endl;

  get_info()->set_start_time(get_info()->get_start_time() - initial_offset_time);
  
  //  cout << "FS: " << ((first_sync-8*channels)/(get_info()->get_npol()))*(8/get_info()->get_nbit())/get_info()->get_rate() << endl;
}


void dsp::Mark4File::initalise ()
{
  
  
  
}

int dsp::Mark4File::count_channels (int file_descriptor) const
{
  // By locating the first sync with only a small number of channels
  // we can proceed to count the number of channels.
  // By definition a BCD cannot have a complete set of 1s.

  int num_chans = 0;
  char cc[4];
  bool sync_completed = false;
  
  uint64_t inital_pos = lseek(file_descriptor, 0, SEEK_CUR);
  
  uint64_t sync_pos = find_sync(file_descriptor);
  
  lseek(file_descriptor,sync_pos,SEEK_SET); // seek to the SYNC
  
  do{
    
    read(file_descriptor, cc, 4); // reads 4 bytes at a time - size of SYNC
    if (cc[0] == SYNC_pattern && cc[1] == SYNC_pattern && cc[2] == SYNC_pattern && cc[3] == SYNC_pattern){
      num_chans ++;
    }else{
      sync_completed = true;
    }
  }while(! sync_completed);
    
  lseek(file_descriptor,inital_pos,SEEK_SET);  // restored original file pointer
  
  return num_chans;
  
}

uint64_t dsp::Mark4File::find_sync (int file_descriptor, uint64_t from) const
{
  // If from is defined it uses that offset, otherwise it defaults to the current position
  // returned number is position relative to current position where the next sync is.
  
  uint64_t next_sync = 0;
  bool sync_found = false;
  char cc[4];
  unsigned int syncs = 0;
  unsigned int search_length = 0;

  uint64_t inital_pos = lseek(file_descriptor, 0, SEEK_CUR);

  if(from !=0)
    lseek(file_descriptor, from, SEEK_CUR);

  do{
    
    read(file_descriptor, cc, 4); // reads 4 bytes at a time - size of SYNC
    if (cc[0] == SYNC_pattern && cc[1] == SYNC_pattern && cc[2] == SYNC_pattern && cc[3] == SYNC_pattern){
      syncs++;
    }else{
      syncs =0;
    }
    
    if(syncs == channels) // located sync -- is this correct?
      sync_found = true;
    
    search_length++;
    if(search_length>2500*channels*10) // *10 is to give a "largish" search area
      return 0;
    
  }while(!sync_found);
  
  next_sync = lseek(file_descriptor, 0, SEEK_CUR) - 4*channels; // 4*channels is because at end of sync
  //  cerr << "FOUND  position: " << currpos << endl;

  lseek(file_descriptor,inital_pos,SEEK_SET);  // restored original file pointer

  return next_sync;
  
}

MJD dsp::Mark4File::decode_date(uint64_t from)
{
  char *timecode = new char[8*channels]; // 8 bytes per channel

  MJD date;
  MJD current;
  utc_t utcdate;
  utc_t tmpdate;
  
  // Special lookup table, see documentation for Mark4 formaters.
  // 0=0, 1=1.25, 2=2.5, 3=3.75, 4=NA, 5=5.0, 6=6.25, 7=7.5, 8=8.75, 9=NA
  //Required for Mark4 standard format
  float time_code_table[] = {0,1.25,2.5,3.75,0.0,5.0,6.25,7.50,8.75,0.0};

  current.Construct(time(NULL));
  
  uint64_t inital_pos = lseek(fd, 0, SEEK_CUR);
  
  uint64_t next_sync = find_sync(fd, from);
  
  // Read the 8 bytes after the SYNC - and handle them as per modes
  lseek(fd,next_sync+4*channels,SEEK_SET);
  
  read(fd, timecode, 8*channels);

  int stepsize = channels/8;

  int julian = 0;
  int year   = 0;
  int day    = 0;
  int hour   = 0;
  int minute = 0;
  double second = 0.0;
  char tmp[4];
  
  switch(mode) {
    
  case VLBA:

    // Date format for VLBA = JJJSSSSS.ssss
    
    // JJJ
    tmp[0] = timecode[0*stepsize];
    tmp[1] = timecode[1*stepsize];
    tmp[2] = timecode[2*stepsize];
    tmp[3] = timecode[3*stepsize];
    julian += decode_bcd(tmp)*100;

    tmp[0] = timecode[4*stepsize];
    tmp[1] = timecode[5*stepsize];
    tmp[2] = timecode[6*stepsize];
    tmp[3] = timecode[7*stepsize];
    julian += decode_bcd(tmp)*10;

    tmp[0] = timecode[8*stepsize];
    tmp[1] = timecode[9*stepsize];
    tmp[2] = timecode[10*stepsize];
    tmp[3] = timecode[11*stepsize];
    julian += decode_bcd(tmp)*1;
    
    //SSSSS
    tmp[0] = timecode[12*stepsize];
    tmp[1] = timecode[13*stepsize];
    tmp[2] = timecode[14*stepsize];
    tmp[3] = timecode[15*stepsize];
    second += decode_bcd(tmp)*10000.0;

    tmp[0] = timecode[16*stepsize];
    tmp[1] = timecode[17*stepsize];
    tmp[2] = timecode[18*stepsize];
    tmp[3] = timecode[19*stepsize];
    second += decode_bcd(tmp)*1000.0;

    tmp[0] = timecode[20*stepsize];
    tmp[1] = timecode[21*stepsize];
    tmp[2] = timecode[22*stepsize];
    tmp[3] = timecode[23*stepsize];
    second += decode_bcd(tmp)*100.0;

    tmp[0] = timecode[24*stepsize];
    tmp[1] = timecode[25*stepsize];
    tmp[2] = timecode[26*stepsize];
    tmp[3] = timecode[27*stepsize];
    second += decode_bcd(tmp)*10.0;

    tmp[0] = timecode[28*stepsize];
    tmp[1] = timecode[29*stepsize];
    tmp[2] = timecode[30*stepsize];
    tmp[3] = timecode[31*stepsize];
    second += decode_bcd(tmp)*1.0;

    //.ssss
    tmp[0] = timecode[32*stepsize];
    tmp[1] = timecode[33*stepsize];
    tmp[2] = timecode[34*stepsize];
    tmp[3] = timecode[35*stepsize];
    second += decode_bcd(tmp)*0.1;

    tmp[0] = timecode[36*stepsize];
    tmp[1] = timecode[37*stepsize];
    tmp[2] = timecode[38*stepsize];
    tmp[3] = timecode[39*stepsize];
    second += decode_bcd(tmp)*0.01;

    tmp[0] = timecode[40*stepsize];
    tmp[1] = timecode[41*stepsize];
    tmp[2] = timecode[42*stepsize];
    tmp[3] = timecode[43*stepsize];
    second += decode_bcd(tmp)*0.001;

    tmp[0] = timecode[44*stepsize];
    tmp[1] = timecode[45*stepsize];
    tmp[2] = timecode[46*stepsize];
    tmp[3] = timecode[47*stepsize];
    second += decode_bcd(tmp)*0.0001;
    
    if(int(current.in_days())%1000 >= julian){
      // 2 most significant digits of 5 digit julian are correct
      julian += ( int(current.in_days())/1000)*1000;
    }
    else{
      julian += ( int(current.in_days())/1000 -1 )*1000;
    }

    date = MJD(julian,int(second), (second-int(second)));
    
    break;
    
  case Standard:
    
    // Date format for VLBA = YDDDHHMMSS.sss
    
    // Y
    tmp[0] = timecode[0*stepsize];
    tmp[1] = timecode[1*stepsize];
    tmp[2] = timecode[2*stepsize];
    tmp[3] = timecode[3*stepsize];
    year = decode_bcd(tmp);


    // DDD
    tmp[0] = timecode[4*stepsize];
    tmp[1] = timecode[5*stepsize];
    tmp[2] = timecode[6*stepsize];
    tmp[3] = timecode[7*stepsize];
    day += decode_bcd(tmp)*100;

    tmp[0] = timecode[8*stepsize];
    tmp[1] = timecode[9*stepsize];
    tmp[2] = timecode[10*stepsize];
    tmp[3] = timecode[11*stepsize];
    day += decode_bcd(tmp)*10;
    
    tmp[0] = timecode[12*stepsize];
    tmp[1] = timecode[13*stepsize];
    tmp[2] = timecode[14*stepsize];
    tmp[3] = timecode[15*stepsize];
    day += decode_bcd(tmp)*1;


    // HH
    tmp[0] = timecode[16*stepsize];
    tmp[1] = timecode[17*stepsize];
    tmp[2] = timecode[18*stepsize];
    tmp[3] = timecode[19*stepsize];
    hour += decode_bcd(tmp)*10;

    tmp[0] = timecode[20*stepsize];
    tmp[1] = timecode[21*stepsize];
    tmp[2] = timecode[22*stepsize];
    tmp[3] = timecode[23*stepsize];
    hour += decode_bcd(tmp)*1;

    // MM
    tmp[0] = timecode[24*stepsize];
    tmp[1] = timecode[25*stepsize];
    tmp[2] = timecode[26*stepsize];
    tmp[3] = timecode[27*stepsize];
    minute += decode_bcd(tmp)*10;

    tmp[0] = timecode[28*stepsize];
    tmp[1] = timecode[29*stepsize];
    tmp[2] = timecode[30*stepsize];
    tmp[3] = timecode[31*stepsize];
    minute += decode_bcd(tmp)*1;


    // SS
    tmp[0] = timecode[32*stepsize];
    tmp[1] = timecode[33*stepsize];
    tmp[2] = timecode[34*stepsize];
    tmp[3] = timecode[35*stepsize];
    second += decode_bcd(tmp)*10.0;

    tmp[0] = timecode[36*stepsize];
    tmp[1] = timecode[37*stepsize];
    tmp[2] = timecode[38*stepsize];
    tmp[3] = timecode[39*stepsize];
    second += decode_bcd(tmp)*1.0;


    // .sss
    tmp[0] = timecode[40*stepsize];
    tmp[1] = timecode[41*stepsize];
    tmp[2] = timecode[42*stepsize];
    tmp[3] = timecode[43*stepsize];
    second += decode_bcd(tmp)*0.1;

    tmp[0] = timecode[44*stepsize];
    tmp[1] = timecode[45*stepsize];
    tmp[2] = timecode[46*stepsize];
    tmp[3] = timecode[47*stepsize];
    second += decode_bcd(tmp)*0.01;

    tmp[0] = timecode[48*stepsize];
    tmp[1] = timecode[49*stepsize];
    tmp[2] = timecode[50*stepsize];
    tmp[3] = timecode[51*stepsize];
    second += time_code_table[decode_bcd(tmp)]*0.001;
    
    
    current.UTC(&tmpdate,0);

    if(tmpdate.tm_year%10 >= year){
      year += int(tmpdate.tm_year/10)*10;
    }else{
      year += int((tmpdate.tm_year/10)-1)*10;
    }
    
    utcdate.tm_year = year;
    utcdate.tm_yday = day;
    utcdate.tm_hour = hour;
    utcdate.tm_min  = minute;
    utcdate.tm_sec  = int(second);
    
    date  = MJD(utcdate);
    date +=  second-int(second); // add fractions of seconds.

    break;
 
   
  default:
    cerr << "Unknown mode - " << mode << endl;
    
  }
  
  //  cerr << "SEEK_CUR: " << lseek(fd,0,SEEK_CUR);
  //  lseek(fd,next_sync-8*channels,SEEK_SET);  // position of decoded time.
  //  cerr << "\t" << lseek(fd,0,SEEK_CUR)/channels << endl; 
  
  lseek(fd,inital_pos,SEEK_SET);  // restored original file pointer
  
  return date;
  
}

int decode_bcd(char in[4]){
  char tmp = ((  in[0] & 0x08) | (in[1] & 0x04) 
	      | (in[2] & 0x02) | (in[3] & 0x01));
  return int(tmp);
}
