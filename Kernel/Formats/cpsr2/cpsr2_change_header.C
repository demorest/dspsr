/*

Quick program by HSK 11 July 2003 to change parameters of a header- cpsr2 or .bs

The 'values' are truncated so that the same number of characters are in the line as for the previous value- so watch out!

*/

#include <string>
#include <vector>
#include <iostream>

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "genutil.h"
#include "Error.h"
#include "Reference.h"
#include "string_utils.h"

#include "dsp/File.h"
#include "dsp/CPSR2File.h"
#include "dsp/BitSeriesFile.h"

const char* args = "f:hk:m:";
void usage();

void parse_it(int argc,char** argv,
	      vector<string>& filenames,
	      vector<string>& keywords, vector<string>& values);

void check_type(dsp::File* loader);
vector<char> get_header(string filename,unsigned header_bytes);
bool match_line(const vector<string>& lines,string keyword, unsigned& the_line);
string chomp_value(string to_chomp, unsigned chars_avail);
void rewrite_header(string filename, vector<char> header);

int main(int argc,char** argv){ try {
  Error::verbose = true;
  dsp::CPSR2File::want_to_yamasaki_verify = false;

  vector<string> filenames;

  vector<string> keywords;
  vector<string> values;

  bool verbose = false;

  parse_it(argc,argv,filenames,keywords,values);

  for(unsigned ifile=0; ifile<filenames.size(); ifile++){
    fprintf(stderr,"Working with file '%s'\n",
	    filenames[ifile].c_str());

    Reference::To<dsp::File> loader(dsp::File::create(filenames[ifile],0));
    loader->open( filenames[ifile] );
    check_type(loader.get());

    vector<char> header = get_header(filenames[ifile],loader->get_header_bytes());

    if( verbose ){
      fprintf(stderr,"\nGot header:\n");
      for( unsigned i=0; i<header.size(); i++)
	fprintf(stderr,"%c",header[i]);

      fprintf(stderr,"header[%d]='%c'\theader[%d]='%d'\n",
	      header.size()-2,header[header.size()-2],header.size()-1,header.back());
    }

    string to_breakup = string(header.begin(),header.end());
    vector<string> lines = stringlines(to_breakup);

    if( verbose ){
      fprintf(stderr,"\nGot %d lines:\n",lines.size());
      for( unsigned i=0; i<lines.size(); i++)
	fprintf(stderr,"%d: '%s'\n",i,lines[i].c_str());
      fprintf(stderr,"\n");
    }

    for( unsigned i=0; i<keywords.size(); i++){
      if( verbose )
	fprintf(stderr,"working with keyword '%s'\n",keywords[i].c_str());

      unsigned the_line = 0;
      if( !match_line(lines,keywords[i],the_line) ){
	fprintf(stderr,"Failed to match keyword '%s' for file '%s'\n",
		keywords[i].c_str(),filenames[i].c_str());
	continue;
      }

      if( verbose )
	fprintf(stderr,"Got the_line='%s'\n",lines[the_line].c_str());

      vector<string> words = stringdecimate(lines[the_line]," \t");

      if( verbose ){
	fprintf(stderr,"Got %d words:\n",words.size());
	for( unsigned iword=0; iword<words.size(); iword++)
	  fprintf(stderr,"%d\t'%s'\n",iword,words[iword].c_str());
      }

      unsigned chars_avail = lines[the_line].size()-words[0].size()-1; 
      string new_value = chomp_value(values[i],chars_avail);

      if( verbose )
	fprintf(stderr,"Got %d chars avail and new_value='%s'\n",chars_avail,new_value.c_str());

      lines[the_line] = words[0] + " " + new_value;

      if( verbose )
	fprintf(stderr,"the_line now '%s'\n",lines[the_line].c_str());
    }

    string str_header = stringdelimit(lines,'\n') + '\n';

    if( verbose )
      fprintf(stderr,"Got str_header (%d vs %d):\n'%s'\n",
	      str_header.size(),loader->get_header_bytes(),str_header.c_str());

    unsigned ichar = 0;
    for( ichar=0; ichar<str_header.size(); ichar++)
      header[ichar] = str_header[ichar];
    for( ;ichar<unsigned(loader->get_header_bytes());ichar++)
      header[ichar] = 0;
      
    rewrite_header(filenames[ifile],header);
  }

} catch(Error& er) {
  cerr << er << endl;
  exit(-1);
} catch( ... ) {
  fprintf(stderr,"Unknown exception caught!\n");
  exit(-1);
}
 printf("Biyee!\n"); 
 exit(0);
}

void check_type(dsp::File* loader){
  dsp::CPSR2File* cpsr2file = dynamic_cast<dsp::CPSR2File*>(loader);
  dsp::BitSeriesFile* bitseriesfile = dynamic_cast<dsp::BitSeriesFile*>(loader);
  
  if( !cpsr2file && !bitseriesfile )
    throw Error(InvalidState,"check_type()",
		"Input files have to be of type CPSR2File or BitSeriesFile\n");
}

vector<char> get_header(string file,unsigned header_bytes){
  int fd = open(file.c_str(),O_RDONLY);

  if (fd < 0) 
    throw Error(FailedCall,"get_header()",
		"failed open(%s): %s",file.c_str(), strerror(errno));
  
  vector<char> header(header_bytes);

  int retval = read (fd, &(header[0]), header_bytes);
  
  close (fd);    

  if (retval < int(header_bytes) )
    throw Error(FailedCall,"get_header()",
		"failed read: %s",strerror(errno));
  
  return header;
}

bool match_line(const vector<string>& lines,string keyword, unsigned& the_line){
  for( unsigned iline=0; iline<lines.size(); iline++){
    if( lines[iline].substr(0,keyword.size())==keyword ){
      the_line = iline;
      return true;
    }
  }
  return false;
}

string chomp_value(string to_chomp, unsigned chars_avail){
  if( to_chomp.size() < chars_avail ){
    to_chomp.insert(to_chomp.begin(),chars_avail-to_chomp.size(),' ');
    return to_chomp;
  }

  if( to_chomp.size() > chars_avail ){
    string oldie = to_chomp;
    to_chomp = string(to_chomp.begin(),to_chomp.begin()+chars_avail);
    fprintf(stderr,"WARNING! Not enough chars available!  '%s' -> '%s'\n",
	    oldie.c_str(),to_chomp.c_str());
  }

  return to_chomp;
}

void rewrite_header(string file,vector<char> header){
  int fd = open(file.c_str(),O_WRONLY);

  if (fd < 0) 
    throw Error(FailedCall,"rewrite_header()",
		"failed open(%s): %s",file.c_str(), strerror(errno));
  
  unsigned bytes_written = write(fd, &(header[0]),header.size());

  close (fd);    

  if (bytes_written < header.size() )
    throw Error(FailedCall,"rewrite_header()",
		"Only wrote %d/%d bytes to header\n",
		bytes_written,header.size());
}

void parse_it(int argc,char** argv,
	      vector<string>& filenames,
	      vector<string>& keywords, vector<string>& values)
{
  if( argc==1 )
    usage();

  int c;

  //const char* args = "f:hk:m:";

  while ((c = getopt(argc, argv, args)) != -1){
    switch (c) {
      
    case 'f':
      filenames.push_back( optarg );
      break;
    case 'h':
      usage();
    case 'k':
      keywords.push_back( optarg );
      values.push_back( argv[optind] );
      fprintf(stderr,"Got a keyword of '%s' and value of '%s'\n",
	      keywords.back().c_str(), values.back().c_str());
      optind++;
      break;
    case 'm':
      parse_metafile(filenames,optarg);
      break;
    default:
      fprintf(stderr,"Could not parse command line.\n");
      exit(-1);
    }
  }

}

void usage(){
  cout << "cpsr2_change_header - by HSK 11 July 2003" << endl;
  cout << "A program to edit the headers of cpsr2 and/or .bs files\n" << endl;
  cout << "Note: for each '-k' call, the keyword, a space(s), and then the value are written out.  If there are not enough chars available in the line, the value is cocatenated.   BEWARE!\n" << endl;
  cout << "Usage: cpsr2_change_header -[" << args << "]" << endl;
  cout << " f filename            Process this filename ('-f' or '-m' req) {fm}\n"
       << " h                     This help page\n"
       << " k keyword value       Replace the line starting with 'keyword' with 'keyword value'\n"
       << " m metafile            Metafile of files to process ('-f' or '-m' req) {fm}\n"
       << endl;

  exit(0);
}
