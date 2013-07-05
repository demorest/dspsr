/***************************************************************************
 *
 *   Copyright (C) 2006 by Eric Plum
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
//! Choose operating system below

#include "dsp/FadcFile.h"
#include "dsp/FadcObservation.h"
#include "fadc_header.h"   
#include "Error.h"
#include <unistd.h>

#include <fstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace std;

// All sampleorders in the following 
// structures have been experimentally determined.
// So they should all be correct!

#ifdef FadcSunOS
typedef struct {
  unsigned t5  :2; 
  unsigned t4  :2; 
  unsigned t7  :2; 
  unsigned t6  :2;
  unsigned t1  :2;
  unsigned t0  :2;
  unsigned t3  :2;
  unsigned t2  :2;
  unsigned t13 :2; 
  unsigned t12 :2; 
  unsigned t15 :2; 
  unsigned t14 :2;  
  unsigned t9  :2;
  unsigned t8  :2;
  unsigned t11 :2;
  unsigned t10 :2;
} two_bit_in_1ADC;

typedef struct {
  unsigned t2  :2;
  unsigned t0  :2;
  unsigned t6  :2;
  unsigned t4  :2;
  unsigned t3  :2; 
  unsigned t1  :2; 
  unsigned t7  :2; 
  unsigned t5  :2;
  unsigned t10 :2;
  unsigned t8  :2;
  unsigned t14 :2;
  unsigned t12 :2;
  unsigned t11 :2; 
  unsigned t9  :2; 
  unsigned t15 :2; 
  unsigned t13 :2;  
} two_bit_in_2ADC;

typedef struct {
  unsigned f3 :4;
  unsigned f2 :4;
  unsigned f1 :4;
  unsigned f0 :4;
  unsigned f7 :4;
  unsigned f6 :4;
  unsigned f5 :4;
  unsigned f4 :4;
} four_bit_in_1ADC;

typedef struct {
  unsigned f2 :4;   // re1
  unsigned f0 :4;   // re0
  unsigned f3 :4;   // im1
  unsigned f1 :4;   // im0
  unsigned f6 :4;   // re3
  unsigned f4 :4;   // re2
  unsigned f7 :4;   // im3
  unsigned f5 :4;   // im2
} four_bit_in_2ADC;
#endif

#ifdef FadcLinux
typedef struct {
  unsigned t6  :2;
  unsigned t7  :2;
  unsigned t4  :2;
  unsigned t5  :2;
  unsigned t2  :2;
  unsigned t3  :2; 
  unsigned t0  :2; 
  unsigned t1  :2; 
  unsigned t14 :2;
  unsigned t15 :2;
  unsigned t12 :2;
  unsigned t13 :2;
  unsigned t10 :2;  
  unsigned t11 :2; 
  unsigned t8  :2; 
  unsigned t9  :2; 
} two_bit_in_1ADC;

typedef struct {
  unsigned t4  :2;
  unsigned t6  :2;
  unsigned t0  :2;
  unsigned t2  :2;
  unsigned t5  :2;
  unsigned t7  :2; 
  unsigned t1  :2; 
  unsigned t3  :2; 
  unsigned t12 :2;
  unsigned t14 :2;
  unsigned t8  :2;
  unsigned t10 :2;
  unsigned t13 :2;  
  unsigned t15 :2; 
  unsigned t9  :2; 
  unsigned t11 :2; 
} two_bit_in_2ADC;

typedef struct {
  unsigned f2 :4;
  unsigned f3 :4;
  unsigned f0 :4;
  unsigned f1 :4;
  unsigned f6 :4;
  unsigned f7 :4;
  unsigned f4 :4;
  unsigned f5 :4;
} four_bit_in_1ADC;

typedef struct {
  unsigned f0 :4;  // re0
  unsigned f2 :4;  // re1
  unsigned f1 :4;  // im0
  unsigned f3 :4;  // im1
  unsigned f4 :4;  // re2
  unsigned f6 :4;  // re3
  unsigned f5 :4;  // im2
  unsigned f7 :4;  // im3
} four_bit_in_2ADC;
#endif




// works for all 8bit cases
typedef struct {
  unsigned e0 :8;
  unsigned e1 :8;
  unsigned e2 :8;
  unsigned e3 :8;
} eight_bit_in;

void dsp::FadcFile::writeByte(FILE* outfile, two_bit_out two)
{
   two.lIm = (unsigned) -((signed int) two.lIm-3);
   two.rIm = (unsigned) -((signed int) two.rIm-3);
   putc((reinterpret_cast<char*> (&two))[0], outfile);
}

void dsp::FadcFile::writeByte(FILE* outfile, four_bit_out four)
{
   four.Im = (unsigned) -((signed int) four.Im-15);
   putc((reinterpret_cast<char*> (&four))[0], outfile);
}

void switch_sign(unsigned char* c)
{
   c[0] = (unsigned char) -((signed int) c -127);
}


dsp::FadcFile::FadcFile (const char* filename) : File ("Fadc")
{
//  string filenameString = filename ? filename : "";
//  string dataFileName = "./Data/" + filenameString + ".000000";
//! Do I need the following lines or can I throw them away ??  
//  if (filename)
//    open(dataFileName.c_str());  
    
//  if (filename) 
//    open (filename);
}

dsp::FadcFile::~FadcFile ( )
{
}

bool dsp::FadcFile::fileExists (char* fileName)
{
   ifstream infile;
   infile.open(fileName);
   bool ret = (infile.is_open());
   infile.close();
   return ret;
}

uint64_t dsp::FadcFile::fileSize(char* fileName)
{
   FILE* testFile;
   uint64_t sz = 0;
   if ((testFile = fopen(fileName, "rb")) == NULL) sz = 0;
   else if (feof(testFile)) sz = 0;
   else
   {
      fseeko(testFile, 0, SEEK_SET);
      uint64_t beg = static_cast <uint64_t> (ftello(testFile));
      fseeko(testFile, 0, SEEK_END);
      sz = (static_cast <uint64_t> (ftello(testFile))) - beg;
   }
   fclose(testFile);
cerr<<"fileSize going to return "<<sz<<"\n";
   return sz;

}


// returns the whole header file
string dsp::FadcFile::get_header (const char* filename)
{
::cerr<<"FadcFile getheader\n";
  string str="";
  char ch;
  std::ifstream input (filename);
  
::cerr<<"Tried to open "<<filename<<" (did not check for success yet)\n";
  if (!input)
    return str;

    while (input.get(ch)) str+=ch; 
  return str;
} 

bool dsp::FadcFile::is_valid (const char* filename) const
{
  string header = get_header (filename);

  if (header.empty())
    return false;

  // verify that the buffer read contains a valid Fadc header
  char hdrstr[64];
  if (fadc_header_get (header.c_str(), "Observatory Name", "%s", hdrstr) < 0)
    return false;     // keyword does not exist
  if (fadc_header_get (header.c_str(), "Digital_Channel No", "%s", hdrstr) < 0)
    return false;     // keyword does not exist
  if (fadc_header_get (header.c_str(), "Sample Res. (Bits)", "%s", hdrstr) < 0)
    return false;     // keyword does not exist
  if (fadc_header_get (header.c_str(), "Decimation_Factor", "%s", hdrstr) < 0)
    return false;     // keyword does not exist
  if (fadc_header_get (header.c_str(), "Number of ADC", "%s", hdrstr) < 0)
    return false;     // keyword does not exist

  return true;
}


int dsp::FadcFile::read_blockMap( long *buffers_per_file, long *bytes_per_buffer)
{
  // read block.map find bytes per buffer and buffers per file
  std::ifstream block_map ("block.map");
  if (!block_map)
  {
    //*buffers_per_file = 4;    changed to use number of ADC per Card from Exp file
    *bytes_per_buffer = 1048576;
    cerr<<"FadcFile: Cannot open block.map, assume "<<(*buffers_per_file)<<" buffers per file, 1048576 bytes per buffer.";
    return -1;
  }
  else 
  {
    const char* line="";
    char ch;
    string str="";
    
    while (block_map.get(ch)) str+=ch; 
    
    line = str.c_str();
    
    if (fadc_blockmap_get(line, "Each File is", "%ld", buffers_per_file) < 0)
    {
      cerr<<"FadcFile", "failed read buffers_per_file from block.map";
      //*buffers_per_file = 4;        changed to use number of ADC per Card from Exp file
    }
 
    if (fadc_blockmap_get(line, "Each Buffer is", "%ld", bytes_per_buffer) < 0)
    {
      cerr<<"FadcFile - failed read bytes_per_buffer from block.map, assume 1048576";
      *bytes_per_buffer = 1048576;
    }
    
    block_map.close();
cerr<<"\nRead block.map\n   buffers_per_file = "<<*buffers_per_file<<"\n   bytes_per_buffer = "<<*bytes_per_buffer<<"\n\n";
  }  // block.map taken care of
  return 0;
}



void dsp::FadcFile::open_file (const char* filename)
{
cerr<<"Entered Fadc open_file, argument: filename = "<<filename<<"\n";  
  string header = get_header (filename);
cerr<<"got header of length = "<<header.length()<<"\n";
  string filenameString = filename;
  
  if (header.empty())
    throw Error (FailedCall, "dsp::FadcFile::open_file",
		 "get_header(%s) failed", filename);

  long firstFile=0, lastFile=0;
  bool okay=true;   // needed to check user input
  long offset_tsmps=0;  // number of timesamples that are cut off the data (first file) as they
                       // don't contain complete data
  long offset_tsmps_file0=0;
  bool use_old_swindata=false;
  
  int nbit;
  if (fadc_header_get (header.c_str(), "Sample Res. (Bits)", "%d", &nbit) < 0)
  throw Error (InvalidState, "FadcFile", "failed read NBIT");
  
  int nADCperCard;
  if (fadc_header_get (header.c_str(), "Number of ADC", "%d", &nADCperCard) < 0)
  throw Error (InvalidState, "FadcFile", "failed read number of ADC per Card");

  long buffers_per_file=nADCperCard;
  long bytes_per_buffer=0;
  read_blockMap(&buffers_per_file, &bytes_per_buffer);
  
  int nChan=0;
  // The number of Cards with ADCs is the same as the number of digital channels
  // count "NumDigital_Channel No" in header
  {
    const char* tmp = header.c_str();
    do 
    {
      tmp=strstr (tmp, "Digital_Channel No");
      if (tmp) 
      {
         nChan++;
	 tmp++;
      }
    } while (tmp);
  }
cerr<<"Found "<<nChan<<" digital channels\n";

  if (buffers_per_file != nChan) cerr<<"FadcFile: Buffers per file ("<<buffers_per_file<<") not equal to # of digital Channels ("<<nChan<<")\n";

  if ((nChan != 1)&&(nChan != 2)&&(nChan != 4)) 
  {
    cerr<<"Found "<<nChan<<" digital channels\n";
    throw Error (InvalidState, "FadcFile", "Unexpected Format: Expected 1, 2 or 4 digital channels");
  }
  
  int nPol=0;
  if ((nADCperCard==1)&&(nChan==4)) nPol=2;    // standard case since August 2003
  else if ((nADCperCard==2)&&(nChan==2)) nPol=2; // standard before August 2003, at that time no magic codes
  else if ((nADCperCard==1)&&(nChan==2)) nPol=1;
  else if ((nADCperCard==2)&&(nChan==1)) nPol=1;
  else if ((nADCperCard==2)&&(nChan==4))
  {
    nPol=2;
    throw Error (InvalidState, "FadcFile", "Unexpected Format: 2pol, 2frequencies(?) has not been implemented yet. [4 digital channels with 2 ADC per card]");
  }
  else
  {
     cerr<<"FadcFile: Unknown data format:   nADCperCard = "<<nADCperCard<<"     nChan = "<<nChan<<"\n";
     throw Error (InvalidState, "Fatal Error (Unknown data format)");
  }
cerr<<"Found "<<nPol<<" polarizations\n";

  // We should expect to find the magic code starting August 2003
  int expect_magic_code = 1;
  {
    char year_mo_str[6];
    for (int i=0;i<6;i++) year_mo_str[i] = filename[i+3];
    long int year_mo = atoi(year_mo_str);
cerr<<"FadcFile: year_mo = "<<year_mo<<"\n";
    if (year_mo < 200308) expect_magic_code = 0;
    else expect_magic_code = 1;
  }
cerr<<"FadcFile: expect_magic_code = "<<expect_magic_code<<"\n";

  // get parameters from last run
  if (fileExists("swindata.info"))
  {
    ifstream lastrun;
    lastrun.open("swindata.info");
    lastrun>>firstFile>>lastFile>>offset_tsmps_file0>>offset_tsmps;
    lastrun.close();
    if (fileExists("swindata"))
    {
       uint64_t predicted_size = (static_cast <uint64_t> (buffers_per_file)) *bytes_per_buffer*(lastFile-firstFile+1)-nbit*nPol*offset_tsmps/4; 
       char answer=' ';
cerr<<"predicted size "<<predicted_size<<'\n';
       if (fileSize("swindata")==predicted_size)
       {
         cout<<"\nFadcFile: Suggestion: process files "<<firstFile<<" through "<<lastFile<<" (faster performance) Y/N : ";
         cin>>answer;
         if (answer=='y'||answer=='Y') use_old_swindata = true;
       }
    }
  }

  if (use_old_swindata == false)  
  do // get numbers of files to process
  {
    okay=true;
    cout<<"\nFadcFile: Please select files for processing";
    cout<<"\nPlease enter number of first file (e.g. 0) : ";
    cin>>firstFile;
    cout<<"\nPlease enter number of last  file (e.g. 0) : ";
    cin>>lastFile;
    if (firstFile<0) 
    {
      cout<<"\nFirst file : Enter a number greater zero, make sure you enter a file number that exists in the Data directory.";
      okay=false;
    }
    if (lastFile<0) 
    {
      cout<<"\nLast file : Enter a number greater zero, make sure you enter a file number that exists in the Data directory.";
      okay=false;
    }
    if (firstFile>lastFile) 
    {
      cout<<"\nFirst file must be smaller than (or equal to) the last file number."
          <<"\nMake sure you enter a file number that exists in the Data directory.";
      okay=false;
    }
  }  
  while(!okay);

  // header is in seperate file
  header_bytes = 0;
    
  // override center frequency or bandwidth, inverted bands
  
  // FREQ  in MHz
  //       This assumes that LCP and RCP have the same center freq (look at "Channel 0")
  double centerFreqOverride=0;
  if (fadc_header_get (header.c_str(), "Center Freq (Hz)", "%lf", &centerFreqOverride) < 0)
    throw Error (InvalidState, "FadcFile", "failed read FREQ");

  cout<<"\nCenter Frequency according to Exp-File: "<<(centerFreqOverride/1000000.)<<" MHz\nPlease enter center frequency (MHz) or 0 to use Exp-File value: ";
  cin>>centerFreqOverride;
  centerFreqOverride*=1000000.;
//! To do: check user input !!

  // BW in MHz
  double expBW=0;
  double expSmpRate=0;
  if (fadc_header_get (header.c_str(), "IF Bandwidth (Hz)", "%lf", &expBW) < 0)
    throw Error (InvalidState, "FadcFile", "failed read BW");
  if (fadc_header_get (header.c_str(), "Sample Rate (Hz)", "%lf", &expSmpRate) < 0)
    throw Error (InvalidState, "FadcFile", "failed read Sampling Rate");
  
  double bwOverride=expSmpRate;  

  cout<<"\nBandwidth according to Exp-File: "<<(expBW/1000000.)<<" MHz\n"
      <<"Sampling rate according to Exp-File: "<<(expSmpRate/1000000.)<<" MHz\n"
      <<"For an inverted band, select negative bandwidth.\n"
      <<"Please enter bandwidth (MHz) or 0 to use Exp-File sampling rate: ";
  cin>>bwOverride;
  bwOverride*=1000000.;
//! To do: check user input !!
  
  
  // create data file that can be read by the unpacker
  // the unpacker gets only the required number of bytes, 
  // so the bytes have to be in the correct order
  // in the measurement the data is sorted by LCP Re, LCP Im, RCP Re, RCP Im
  // sets offset_tsmps
  if (!use_old_swindata)
  {

     if (0 > createDataFile((char*) filenameString.c_str(), firstFile, lastFile, &offset_tsmps_file0, &offset_tsmps, nbit, nPol, nChan, nADCperCard, buffers_per_file, bytes_per_buffer, expect_magic_code))
        throw Error (InvalidState, "FadcFile", "An error occured in createDataFile, see above");
   }

  FadcObservation data (header.c_str(), firstFile, lastFile, offset_tsmps_file0, offset_tsmps, centerFreqOverride, bwOverride);
  info = data;
cerr<<"FadcFile Initialized info\n";  
  
  
  // open the file
cerr<<"Trying to open swindata\n";
    // :: means in the global namespace
//  fd = ::open ("swindata", O_RDONLY);
  fd = ::open ("swindata", O_RDONLY);
cerr<<"fd = "<<fd<<'\n';
  if (fd < 0)
    throw Error (FailedSys, "dsp::FadcFile::open_file()", 
		 "open(%s) failed", filename);

lseek(fd, 0, SEEK_SET); // This should not be necessary

cerr<<"Datafile open\n";  
  // cannot load less than 4 bytes. set the time sample resolution accordingly
  // resolution = smallest number of time samples that can be loaded at once
  // here: 2bit data, so 4samples per byte, so resolution == 4*4
  //! this may have to be larger (n7dis unpacker works with 16 samples at once)
  resolution = 32 / get_info()->get_nbit();   
  if (resolution == 0)  // should never happen
    resolution = 1;

  if (verbose)
    cerr << "FadcFile::open exit" << endl;
}


int dsp::FadcFile::createDataFile(char* expFileName, long firstFile, long lastFile, long* offset_tsmps_file0, long* offset_tsmps, int nbit, int nPol, int nChan, int nADCperCard, long buffers_per_file, long bytes_per_buffer, int expect_magic_code)
{
  long currentFile = firstFile;

  char* pathData = "./Data";
  
  FILE* infile;
  FILE* outfile;

  // create and open file for output
  outfile = fopen("swindata", "wb");

  if (outfile==NULL) 
  {
     cerr<<"FadcFile: Error: Cannot open output file swindata\n"; 
     return -1; 
  }

  // Test whether enough data is at standard path
  //! THIS SHOULD ALSO BE DONE IN OBSERVATION (needs file 0 in special cases)
   {
       string test, path; 
       char num[6];
       sprintf(num, "%06ld", lastFile);
       test = (string) pathData + "/" + expFileName + "." + num;
cerr<<"\nLooking for "<<test;
       while (!fileExists((char*) test.c_str()) && (((string) pathData)!="exit" ))
       {
          cout<<"\n\nLooking for file "<<lastFile<<": Either none or insufficient data have been found in\n"<<pathData;
	  cout<<"\nType \"exit\" to quit";
	  cout<<"\nPlease enter path to Data files directory without trailing \"/\" : ";
	  cin>>path;
	  pathData = (char*) path.c_str();
          test = (string) pathData + "/" + expFileName + "." + num;
       }

      if (((string) pathData)=="exit") return -1;
      
cerr<<"\nFound "<<test<<"\n\n";
   }

  char* byteDataIn = new char[buffers_per_file*bytes_per_buffer];
             
// read open and read data file
  char* dataFileName = (char*) ((string) pathData + "/" + expFileName + ".000000").c_str();
  
// make sure that datafiles have the correct length
  uint64_t expected_size = (uint64_t) buffers_per_file*bytes_per_buffer;
  uint64_t measured_size = fileSize(dataFileName);
  if (expected_size == measured_size) cerr<<"Datafiles have the correct size: "<<expected_size<<" bytes\n";
  else 
  {
    cerr<<"Datafiles have the wrong size. Expected: "<<expected_size<<" bytes. Found "<<measured_size<<" bytes. Check values in block.map.\n"
           <<"I read (or guessed if I could not read block.map)   buffers per file: "<<buffers_per_file<<"     bytes per buffer: "<<bytes_per_buffer<<"\n";
    return -1;
  }
  
  infile=fopen(dataFileName, "r");
  if (infile==NULL) 
  {
     cerr<<"FadcFile: Error: Cannot open dataFile number 0 for finding magic code offset\n"; 
     return -1; 
  }
  if (fread(byteDataIn, bytes_per_buffer, buffers_per_file, infile)!= (unsigned long) buffers_per_file)
  {
     cerr<<"FadcFile: Error reading data!! (Tried to read file 0 to find magic code offsets.) \n";
     return -1;
  }
  
  //! Find Magic code aa aa 01 00 within the first 1000 or so eligible characters
  //! or aa aa 00 00
  int offsetLimit = (expect_magic_code==1) ? 1000 : 100;
  long* offset= new long[nChan];
  for (int i=0;i<nChan;i++)
     for (offset[i]=0; !((byteDataIn[0+offset[i]+i*bytes_per_buffer]==(char)0xAA)
                      && (byteDataIn[1+offset[i]+i*bytes_per_buffer]==(char)0xAA)
		      && ((byteDataIn[2+offset[i]+i*bytes_per_buffer]==(char)0x01)||(byteDataIn[2+offset[i]+i*bytes_per_buffer]==(char)0x00))
		      && (byteDataIn[3+offset[i]+i*bytes_per_buffer]==(char)0x00) 
		     || offset[i]>offsetLimit)     ;offset[i]++);

  for (int i=0;i<nChan;i++)
  {
    offset[i]+=4;  // jump behind magic code
    if (offset[i]>offsetLimit)
    {
      if (expect_magic_code==1)
      {
        cerr<<"Magic Code "<<i<<" is missing. Please enter offset value : \n";
        cin>>offset[i];
       }
       else 
      {
        offset[i]=0;
        cerr<<"Magic Code not expected. Magic Code "<<i<<" not found (as expected). Assumed to be 0.\n";	 
      }
    }
cerr<<i<<"th Magic code offset is "<<offset[i]<<" bytes.\n";
  }
  
  //! fill beginning of file with some zero-value up to largest offset
  //! shift valuable data back and remember bytes that "fall off the edge"
  
  long max_offset = 0;
  for (int i=0;i<nChan;i++) if (max_offset<offset[i]) max_offset=offset[i];
  
cerr<<"Maximum offset is "<<max_offset<<" bytes\n";

  long* shift=new long[nChan];
  for (int i=0;i<nChan;i++) shift[i] = max_offset - offset[i];

  long max_shift = 0;
  for (int i=0;i<nChan;i++) if (max_shift<shift[i]) max_shift=shift[i];
  
cerr<<"Maximum shift is "<<max_shift<<" bytes\n";
  
  // set no of bytes we want to skip from the beginning of each buffer in first file
  // first buffergroup
  long offset_bytes = (firstFile==0) ? max_offset : max_shift;
  *offset_tsmps = 8 * offset_bytes / (nbit * nADCperCard);
  
  *offset_tsmps_file0 = 8 * max_offset / (nbit * nADCperCard);
  
  char remember_new[nChan][max_shift];  // for remembering characters that fall off the edge for next file
  char remember_old[nChan][max_shift];  // for remembering characters that fall off the edge for next file
  for (int i=0;i<nChan;i++)         // initialize remember (should be clearly defined, but it wont be used)
     for (long j=0;j<shift[i];j++)
        remember_old[i][j] = (char) 0x55;

  fclose(infile);  
  

  // DataIn: array of 4byte sized structures containing 16/8/4 unsigned 2bit/4bit/8bit values (16/8/4 samples of one polarisation, re or im)
  // if nADCperCard == 2 then we have alternatin re im values of one polarization in each structure
    
  //! Do the following stuff for every file
  for(currentFile = firstFile; currentFile<=lastFile; currentFile++)
  {
    // open and read data file
    sprintf(dataFileName, "%s/%s.%06ld", pathData, expFileName, currentFile);
cerr<<"Decoding file: "<<dataFileName<<"\n";
    infile=fopen(dataFileName, "rb");
    if (infile==NULL) 
    {
      cerr<<"Cannot open dataFile\n"; 
      return -1; 
    }
    if (fread(byteDataIn, bytes_per_buffer, buffers_per_file, infile)!= (unsigned long) buffers_per_file)
    {
       cerr<<"Error reading data!!\n";
       return -1;
    }
    for (int i=0;i<nChan;i++)     // remembering
       for (long j=0;j<shift[i];j++)
          remember_new[i][j] = byteDataIn[(i+1)*bytes_per_buffer - shift[i] + j];
	
    for (int i=0;i<nChan;i++)           // shift data shift[i] bytes forwards
       for (signed long j=bytes_per_buffer-1; j>=shift[i];j--)
          byteDataIn[bytes_per_buffer*i+j]=byteDataIn[bytes_per_buffer*i+j-shift[i]];

    if (currentFile!=firstFile)   // otherwise we throw away nonsense or incomplete samples at the beginning (later)
      for (int i=0;i<nChan;i++)           // fill in remembered bytes from last file
        for (long j=0;j<shift[i];j++) byteDataIn[i*bytes_per_buffer+j]= remember_old[i][j];
  
    for (int i=0;i<nChan;i++)            // set remember for the next file
      for (long j=0;j<shift[i];j++) 
         remember_old[i][j] = remember_new[i][j]; 
	 
    // //////////////////////////////////////////////////////////////////////
    //                          2 bit 
    // //////////////////////////////////////////////////////////////////////
    
    if ((nbit==2)&&(nADCperCard==1)&&(nChan==4)) // nPol == 2
    {
      two_bit_out outbyte;     

      long fourbytes_per_buffer = bytes_per_buffer/4;
      two_bit_in_1ADC* twoBitDataIn = (two_bit_in_1ADC*) byteDataIn;
  
      // the data consists of groups of 4 buffers
      // 1: LCP real, 2:LCP imag, 3: RCP real, 4:RCP imag
      for (long buffergroup = 0; buffergroup < buffers_per_file/nChan; buffergroup++)
      {
         // skip bytes that don't contain information (at the beginning of the first file)
         long startByte = (firstFile==currentFile && buffergroup==0) ? offset_bytes : 0 ; 
         for (long fourbyte = 0; fourbyte < fourbytes_per_buffer; fourbyte++)  // steps of a 4 bytes = steps of 16 samples
         {
           // read 2bit values sort them LCP_re, LCP_im, RCP_re, RCP_im, next sample, write to file
           if (startByte <= fourbyte*4 + 0)
           {
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t0;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t0;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t0;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t0;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t1;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t1;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t1;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t1;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t2;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t2;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t2;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t2;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t3;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t3;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t3;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t3;  // RCP im
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 1)
           {
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t4;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t4;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t4;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t4;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t5;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t5;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t5;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t5;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t6;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t6;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t6;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t6;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t7;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t7;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t7;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t7;  // RCP im
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 2)
           {  
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t8;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t8;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t8;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t8;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t9;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t9;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t9;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t9;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t10;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t10;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t10;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t10;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t11;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t11;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t11;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t11;  // RCP im
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 3)
           {
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t12;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t12;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t12;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t12;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t13;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t13;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t13;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t13;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t14;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t14;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t14;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t14;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t15;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t15;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].t15;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].t15;  // RCP im
            writeByte(outfile, outbyte);
           }
         }  // end of loop over bytes in a buffer (covers all bytes in a buffergroup of 4 buffers)
      }  // end of loop over buffergroups in a file
    }  // end of 2bit 

// --------------------------------------------------------------
    
    else if ((nbit==2)&&(nADCperCard==2)&&(nChan==2)) // nPol == 2
    {
      two_bit_out outbyte;     

      long fourbytes_per_buffer = bytes_per_buffer/4;
      two_bit_in_2ADC* twoBitDataIn = (two_bit_in_2ADC*) byteDataIn;
  
      // the data consists of groups of 2 buffers
      // 1: LCP real, imag,  2: RCP real, imag
      for (long buffergroup = 0; buffergroup < buffers_per_file/nChan; buffergroup++)
      {
         // skip bytes that don't contain information (at the beginning of the first file)
         long startByte = (firstFile==currentFile && buffergroup==0) ? offset_bytes : 0 ; 
         for (long fourbyte = 0; fourbyte < fourbytes_per_buffer; fourbyte++)  // steps of a 4 bytes = steps of 16 samples
         {
           // read 2bit values sort them LCP_re, LCP_im, RCP_re, RCP_im, next sample, write to file
           if (startByte <= fourbyte*4 + 0)
           {
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t0;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t1;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t0;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t1;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t2;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t3;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t2;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t3;  // RCP im
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 1)
           {
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t4;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t5;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t4;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t5;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t6;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t7;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t6;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t7;  // RCP im
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 2)
           {  
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t8;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t9;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t8;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t9;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t10;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t11;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t10;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t11;  // RCP im
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 3)
           {
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t12;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t13;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t12;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t13;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t14;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t15;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t14;  // RCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t15;  // RCP im
            writeByte(outfile, outbyte);
           }
         }  // end of loop over bytes in a buffer (covers all bytes in a buffergroup of 4 buffers)
      }  // end of loop over buffergroups in a file
    }  // end of 2bit 
    
// --------------------------------------------------------------
    
    else if ((nbit==2)&&(nADCperCard==1)&&(nChan==2)) // nPol == 1
    {
      two_bit_out outbyte;     

      long fourbytes_per_buffer = bytes_per_buffer/4;
      two_bit_in_1ADC* twoBitDataIn = (two_bit_in_1ADC*) byteDataIn;
  
      // the data consists of groups of 2 buffers
      // 1: LCP real, 2:LCP imag   (or other polarization)
      for (long buffergroup = 0; buffergroup < buffers_per_file/nChan; buffergroup++)
      {
         // skip bytes that don't contain information (at the beginning of the first file)
         long startByte = (firstFile==currentFile && buffergroup==0) ? offset_bytes : 0 ; 
         for (long fourbyte = 0; fourbyte < fourbytes_per_buffer; fourbyte++)  // steps of a 4 bytes = steps of 8 samples (16 values)
         {
           // read 2bit values sort them LCP_re, LCP_im, next sample, write to file
           if (startByte <= fourbyte*4 + 0)
           {
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t0;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t0;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t1;  // LCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t1;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t2;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t2;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t3;  // LCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t3;  // LCP im   
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 1)
           {
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t4;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t4;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t5;  // LCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t5;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t6;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t6;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t7;  // LCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t7;  // LCP im   
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 2)
           {  
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t8;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t8;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t9;  // LCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t9;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t10;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t10;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t11;  // LCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t11;  // LCP im   
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 3)
           {
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t12;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t12;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t13;  // LCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t13;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t14;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t14;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t15;  // LCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].t15;  // LCP im   
            writeByte(outfile, outbyte);
           }
         }  // end of loop over bytes in a buffer (covers all bytes in a buffergroup of 4 buffers)
      }  // end of loop over buffergroups in a file
    }  // end of 2bit 

    
// ----------------------------------------------------

    else if ((nbit==2)&&(nADCperCard==2)&&(nChan==1)) // nPol == 1
    {
      two_bit_out outbyte;     

      long fourbytes_per_buffer = bytes_per_buffer/4;
      two_bit_in_2ADC* twoBitDataIn = (two_bit_in_2ADC*) byteDataIn;
  
      // the data consists of groups of 1 buffer
      // 1: LCP real, imag
      for (long buffergroup = 0; buffergroup < buffers_per_file/nChan; buffergroup++)
      {
         // skip bytes that don't contain information (at the beginning of the first file)
         long startByte = (firstFile==currentFile && buffergroup==0) ? offset_bytes : 0 ; 
         for (long fourbyte = 0; fourbyte < fourbytes_per_buffer; fourbyte++)  // steps of a 4 bytes = steps of 8 samples (16 values)
         {
           // read 2bit values sort them LCP_re (or other polarization), next sample, write to file
           if (startByte <= fourbyte*4 + 0)
           {
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t0;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t1;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t2;  // LCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t3;  // LCP im   
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 1)
           {
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t4;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t5;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t6;  // LCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t7;  // LCP im   
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 2)
           {  
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t8;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t9;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t10;  // LCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t11;  // LCP im   
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 3)
           {
             outbyte.lRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t12;  // LCP re
             outbyte.lIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t13;  // LCP im   
             outbyte.rRe = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t14;  // LCP re
             outbyte.rIm = twoBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].t15;  // LCP im   
            writeByte(outfile, outbyte);
           }
         }  // end of loop over bytes in a buffer (covers all bytes in a buffergroup of 4 buffers)
      }  // end of loop over buffergroups in a file
    }  // end of 2bit 
    
        
    // //////////////////////////////////////////////////////////////////////
    //                          4 bit 
    // //////////////////////////////////////////////////////////////////////
    
    else if ((nbit==4)&&(nADCperCard==1)&&(nChan==4)) // nPol == 2
    {
      four_bit_out outbyte;     

      long fourbytes_per_buffer = bytes_per_buffer/4;
      four_bit_in_1ADC* fourBitDataIn = (four_bit_in_1ADC*) byteDataIn;
  
      // the data consists of groups of 4 buffers
      // 1: LCP real, 2:LCP imag, 3: RCP real, 4:RCP imag
      for (long buffergroup = 0; buffergroup < buffers_per_file/nChan; buffergroup++)
      {
         // skip bytes that don't contain information (at the beginning of the first file)
         long startByte = (firstFile==currentFile && buffergroup==0) ? offset_bytes : 0 ; 
         for (long fourbyte = 0; fourbyte < fourbytes_per_buffer; fourbyte++)  // steps of a 4 bytes = steps of 8 samples
         {
           // read 4bit values sort them LCP_re, LCP_im, RCP_re, RCP_im, next sample, write to file
           if (startByte <= fourbyte*4 + 0)
           {
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f0;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f0;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].f0;  // RCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].f0;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f1;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f1;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].f1;  // RCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].f1;  // RCP im
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 1)
           {
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f2;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f2;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].f2;  // RCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].f2;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f3;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f3;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].f3;  // RCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].f3;  // RCP im
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 2)
           {  
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f4;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f4;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].f4;  // RCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].f4;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f5;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f5;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].f5;  // RCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].f5;  // RCP im
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 3)
           {
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f6;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f6;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].f6;  // RCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].f6;  // RCP im
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f7;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f7;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].f7;  // RCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].f7;  // RCP im
            writeByte(outfile, outbyte);
           }
         }  // end of loop over bytes in a buffer (covers all bytes in a buffergroup of 4 buffers)
      }  // end of loop over buffergroups in a file
    }  // end of 4bit 

// -----------------------------------------------------

    else if ((nbit==4)&&(nADCperCard==2)&&(nChan==2)) // nPol == 2
    {
      four_bit_out outbyte;     

      long fourbytes_per_buffer = bytes_per_buffer/4;
      four_bit_in_2ADC* fourBitDataIn = (four_bit_in_2ADC*) byteDataIn;
  
//ofstream ftest;
//ftest.open("swin1"); 
      // the data consists of groups of 2 buffers
      // 1: LCP real, imag,   2: RCP real, imag
      for (long buffergroup = 0; buffergroup < buffers_per_file/nChan; buffergroup++)
      {
         // skip bytes that don't contain information (at the beginning of the first file)
         long startByte = (firstFile==currentFile && buffergroup==0) ? offset_bytes : 0 ; 
         for (long fourbyte = 0; fourbyte < fourbytes_per_buffer; fourbyte++)  // steps of a 4 bytes = steps of 8 values (4 samples)
         {
           // read 4bit values sort them LCP_re, LCP_im, RCP_re, RCP_im, next sample, write to file

           if (startByte <= fourbyte*4 + 0)
           {
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f0;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f1;  // LCP im
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f0;  // RCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f1;  // RCP im
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 1)
           {
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f2;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f3;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f2;  // RCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f3;  // RCP im
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 2)
           {  
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f4;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f5;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f4;  // RCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f5;  // RCP im
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 3)
           {
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f6;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f7;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f6;  // RCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f7;  // RCP im
            writeByte(outfile, outbyte);
           }
         }  // end of loop over bytes in a buffer (covers all bytes in a buffergroup of 4 buffers)
      }  // end of loop over buffergroups in a file
    }  // end of 4bit 
    
// --------------------------------------------------------------

    else if ((nbit==4)&&(nADCperCard==1)&&(nChan==2)) // nPol == 1
    {
      four_bit_out outbyte;     

      long fourbytes_per_buffer = bytes_per_buffer/4;
      four_bit_in_1ADC* fourBitDataIn = (four_bit_in_1ADC*) byteDataIn;
  
      // the data consists of groups of 2 buffers
      // 1: LCP real, 2:LCP imag (or other polarization)
      for (long buffergroup = 0; buffergroup < buffers_per_file/nChan; buffergroup++)
      {
         // skip bytes that don't contain information (at the beginning of the first file)
         long startByte = (firstFile==currentFile && buffergroup==0) ? offset_bytes : 0 ; 
         for (long fourbyte = 0; fourbyte < fourbytes_per_buffer; fourbyte++)  // steps of a 4 bytes = steps of 8 samples
         {
           // read 4bit values sort them LCP_re, LCP_im, RCP_re, RCP_im, next sample, write to file
           if (startByte <= fourbyte*4 + 0)
           {
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f0;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f0;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f1;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f1;  // LCP im   
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 1)
           {
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f2;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f2;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f3;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f3;  // LCP im   
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 2)
           {  
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f4;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f4;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f5;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f5;  // LCP im   
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 3)
           {
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f6;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f6;  // LCP im   
            writeByte(outfile, outbyte);
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f7;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].f7;  // LCP im   
            writeByte(outfile, outbyte);
           }
         }  // end of loop over bytes in a buffer (covers all bytes in a buffergroup of 4 buffers)
      }  // end of loop over buffergroups in a file
    }  // end of 4bit 

// ------------------------------------------------------

    else if ((nbit==4)&&(nADCperCard==2)&&(nChan==1)) // nPol == 1
    {
      four_bit_out outbyte;     

      long fourbytes_per_buffer = bytes_per_buffer/4;
      four_bit_in_2ADC* fourBitDataIn = (four_bit_in_2ADC*) byteDataIn;
  
      // the data consists of groups of 1 buffer
      // 1: LCP real, imag
      for (long buffergroup = 0; buffergroup < buffers_per_file/nChan; buffergroup++)
      {
         // skip bytes that don't contain information (at the beginning of the first file)
         long startByte = (firstFile==currentFile && buffergroup==0) ? offset_bytes : 0 ; 
         for (long fourbyte = 0; fourbyte < fourbytes_per_buffer; fourbyte++)  // steps of a 4 bytes = steps of 8 samples
         {
           // read 4bit values sort them LCP_re, LCP_im (or other polarization), next sample, write to file
           if (startByte <= fourbyte*4 + 0)
           {
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f0;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f1;  // LCP im   
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 1)
           {
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f2;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f3;  // LCP im
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 2)
           {  
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f4;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f5;  // LCP im   
            writeByte(outfile, outbyte);
           }

           if (startByte <= fourbyte*4 + 3)
           {
             outbyte.Re = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f6;  // LCP re
             outbyte.Im = fourBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].f7;  // LCP im
            writeByte(outfile, outbyte);
           }
         }  // end of loop over bytes in a buffer (covers all bytes in a buffergroup of 4 buffers)
      }  // end of loop over buffergroups in a file
    }  // end of 4bit 
        
    // //////////////////////////////////////////////////////////////////////
    //                          8 bit 
    // //////////////////////////////////////////////////////////////////////
    // NOTE : This could be written much simpler by just dealing with chars
    
    else if ((nbit==8)&&(nADCperCard==1)&&(nChan==4)) // nPol == 2
    {
      unsigned char outbyte;
      
      long fourbytes_per_buffer = bytes_per_buffer/4;
      eight_bit_in* eightBitDataIn = (eight_bit_in*) byteDataIn;
  
      // the data consists of groups of 4 buffers
      // 1: LCP real, 2:LCP imag, 3: RCP real, 4:RCP imag
      for (long buffergroup = 0; buffergroup < buffers_per_file/nChan; buffergroup++)
      {
         // skip bytes that don't contain information (at the beginning of the first file)
         long startByte = (firstFile==currentFile && buffergroup==0) ? offset_bytes : 0 ; 
         for (long fourbyte = 0; fourbyte < fourbytes_per_buffer; fourbyte++)  // steps of 4 bytes = steps of 4 samples
         {
           // read 8bit values sort them LCP_re, LCP_im, RCP_re, RCP_im, next sample, write to file
           if (startByte <= fourbyte*4 + 0)
           {
             outbyte = eightBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].e0;  // LCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].e0;  // LCP im   
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].e0;  // RCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].e0;  // RCP im
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
           }

           if (startByte <= fourbyte*4 + 1)
           {
             outbyte = eightBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].e1;  // LCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].e1;  // LCP im   
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].e1;  // RCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].e1;  // RCP im
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
           }

           if (startByte <= fourbyte*4 + 2)
           {  
             outbyte = eightBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].e2;  // LCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].e2;  // LCP im   
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].e2;  // RCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].e2;  // RCP im
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
           }

           if (startByte <= fourbyte*4 + 3)
           {
             outbyte = eightBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].e3;  // LCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].e3;  // LCP im   
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].e3;  // RCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].e3;  // RCP im
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
           }
         }  // end of loop over bytes in a buffer (covers all bytes in a buffergroup of 4 buffers)
      }  // end of loop over buffergroups in a file
    }  // end of 8bit 
    
// --------------------------------------------------

    else if ((nbit==8)&&(nADCperCard==2)&&(nChan==2)) // nPol == 2
    {
      unsigned char outbyte;
      
      long fourbytes_per_buffer = bytes_per_buffer/4;
      eight_bit_in* eightBitDataIn = (eight_bit_in*) byteDataIn;
  
      // the data consists of groups of 2 buffers
      // 1: LCP real, imag,    2: RCP real, imag
      for (long buffergroup = 0; buffergroup < buffers_per_file/nChan; buffergroup++)
      {
         // skip bytes that don't contain information (at the beginning of the first file)
         long startByte = (firstFile==currentFile && buffergroup==0) ? offset_bytes : 0 ; 
	 
	 if (startByte%2 != 0) 
	 {
	   cerr<<"FadcFile: Trying to write swindata for 8bit, 2ADCperCard, 2Digital Channels (2 Polarizations):\n"
	       <<"    Fatal Error: startByte not divisible by 2.\n"
	       <<"     startByte is determined by the magic code.\n";
	   return -1;
	 }
	 
         for (long fourbyte = 0; fourbyte < fourbytes_per_buffer; fourbyte++)  // steps of a 4 bytes = steps of 4 values (2 samples)
         {
           // read 8bit values sort them LCP_re, LCP_im, RCP_re, RCP_im, next sample, write to file
           if (startByte <= fourbyte*4 + 0)
           {
             outbyte = eightBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].e0;  // LCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].e1;  // LCP im   
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].e0;  // RCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].e1;  // RCP im
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
           }

           if (startByte <= fourbyte*4 + 2)
           {  
             outbyte = eightBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].e2;  // LCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 2)*fourbytes_per_buffer + fourbyte].e3;  // LCP im   
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].e2;  // RCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 3)*fourbytes_per_buffer + fourbyte].e3;  // RCP im
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
           }

         }  // end of loop over bytes in a buffer (covers all bytes in a buffergroup of 4 buffers)
      }  // end of loop over buffergroups in a file
    }  // end of 8bit 
        
// ------------------------------------------------------------    

    else if ((nbit==8)&&(nADCperCard==1)&&(nChan==2)) // nPol == 1
    {
      unsigned char outbyte;
      
      long fourbytes_per_buffer = bytes_per_buffer/4;
      eight_bit_in* eightBitDataIn = (eight_bit_in*) byteDataIn;
  
      // the data consists of groups of 2 buffers
      // 1: LCP real, 2:LCP imag
      for (long buffergroup = 0; buffergroup < buffers_per_file/nChan; buffergroup++)
      {
         // skip bytes that don't contain information (at the beginning of the first file)
         long startByte = (firstFile==currentFile && buffergroup==0) ? offset_bytes : 0 ; 
         for (long fourbyte = 0; fourbyte < fourbytes_per_buffer; fourbyte++)  // steps of  4 bytes = steps of 4 samples
         {
           // read 8bit values sort them LCP_re, LCP_im (or other polarization), next sample, write to file
           if (startByte <= fourbyte*4 + 0)
           {
             outbyte = eightBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].e0;  // LCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].e0;  // LCP im   
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
           }

           if (startByte <= fourbyte*4 + 1)
           {
             outbyte = eightBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].e1;  // LCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].e1;  // LCP im   
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
           }

           if (startByte <= fourbyte*4 + 2)
           {  
             outbyte = eightBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].e2;  // LCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].e2;  // LCP im   
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
           }

           if (startByte <= fourbyte*4 + 3)
           {
             outbyte = eightBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].e3;  // LCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
             outbyte = eightBitDataIn[(nChan*buffergroup + 1)*fourbytes_per_buffer + fourbyte].e3;  // LCP im   
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
           }
         }  // end of loop over bytes in a buffer (covers all bytes in a buffergroup of 4 buffers)
      }  // end of loop over buffergroups in a file
    }  // end of 8bit 
    
// ------------------------------------------------------------

    else if ((nbit==8)&&(nADCperCard==2)&&(nChan==1)) // nPol == 1
    {
      unsigned char outbyte;
      
      long fourbytes_per_buffer = bytes_per_buffer/4;
      eight_bit_in* eightBitDataIn = (eight_bit_in*) byteDataIn;
  
      // the data consists of groups of 1 buffer
      // 1: LCP real, imag
      for (long buffergroup = 0; buffergroup < buffers_per_file/nChan; buffergroup++)
      {
         // skip bytes that don't contain information (at the beginning of the first file)
         long startByte = (firstFile==currentFile && buffergroup==0) ? offset_bytes : 0 ; 
         
	 if (startByte%2 != 0) 
	 {
	   cerr<<"FadcFile: Trying to write swindata for 8 bit, 2 ADCperCard, 1 Digital Channel (2 Polarizations):\n"
	       <<"    Fatal Error: startByte not divisible by 2.\n"
	       <<"     startByte is determined by the magic code.\n";
	   return -1;
	 }
	 
	 for (long fourbyte = 0; fourbyte < fourbytes_per_buffer; fourbyte++)  // steps of 4 bytes = steps of 4 values (2 samples)
         {
           // read 8bit values sort them LCP_re, LCP_im, RCP_re, RCP_im, next sample, write to file
           if (startByte <= fourbyte*4 + 0)
           {
             outbyte = eightBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].e0;  // LCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
           }

           if (startByte <= fourbyte*4 + 1)
           {
             outbyte = eightBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].e1;  // LCP im   
	     switch_sign(&outbyte);
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
           }

           if (startByte <= fourbyte*4 + 2)
           {  
             outbyte = eightBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].e2;  // LCP re
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
           }

           if (startByte <= fourbyte*4 + 3)
           {
             outbyte = eightBitDataIn[(nChan*buffergroup + 0)*fourbytes_per_buffer + fourbyte].e3;  // LCP im
	     switch_sign(&outbyte);                     
             putc((reinterpret_cast<char*> (&outbyte))[0], outfile);
           }
         }  // end of loop over bytes in a buffer (covers all bytes in a buffergroup of 4 buffers)
      }  // end of loop over buffergroups in a file
    }  // end of 8bit 

    else 
    {
       cerr<<"FadcFile: Unknown data format: nbit = "<<nbit<<"     nADCperCard = "<<nADCperCard<<"     nChan = "<<nChan<<"\n";
       return -1;
    }
    
// -------------------------------------------------------------
        
    fclose(infile);
  } //end of loop over files
  
  fclose(outfile);
  
  ofstream outinfo;
  outget_info()->open("swindata.info");
  if (outget_info()->is_open())
  {
     outinfo<<firstFile<<' '<<lastFile
            <<"\n"<<*offset_tsmps_file0<<' '<<*offset_tsmps
            <<"\nThe first line contains the number of the first and the last files"
	    <<"\nfrom the Data directory which have been encoded in swindata."
	    <<"\nIt is needed to allow the FadcFile class of the"
	    <<"\nFadc Unpacker (Swinburne software) not to rewrite swindata,"
	    <<"\nif this step is not necessary"
	    <<"\n\nThe 2nd line contains the number of time samples in the file 0"
	    <<"\nand the first file, after which the data encoded in swindata starts";
  }
  else cerr<<"FadcFile: Error (over-) writing swindata.info - cannot open file\n";
  
  outget_info()->close();
  
cerr<<"FadcFile: finished writing swindata \n\n";

  return 0;

}
