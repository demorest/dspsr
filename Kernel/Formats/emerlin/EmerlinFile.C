
#include <iostream>
#include <cstdio>
#include <cstring>
#include <fcntl.h>

#include "dsp/EmerlinFile.h"
#include "dsp/ASCIIObservation.h"
#include "vdifio.h"
#include "ascii_header.h"
#include <unistd.h>



using namespace std;

dsp::EmerlinFile::EmerlinFile(const char* filename, const char* headername) : File("emerlin"),dropped(0) {
}

dsp::EmerlinFile::~EmerlinFile() {
}


bool dsp::EmerlinFile::is_valid(const char* filename) const {
    FILE *fptr = fopen(filename, "r");
    if (!fptr)
    {
        if (verbose)
            cerr << "dsp::EmerlinFile::is_valid Error opening file." << endl;
        return false;
    }

    char header[4096];
    fread(header, sizeof(char), 4096, fptr);
    fclose(fptr);

    char inst[64];
    if ( ascii_header_get(header, "INSTRUMENT", "%s", inst) < 0 )
    {
        if (verbose)
            cerr << "dsp::EmerlinFile::is_valid no INSTRUMENT line" << endl;
        return false;
    }
    if ( std::string(inst) != "EMERLIN" )
    {
        if (verbose)
            cerr << "dsp::EmerlinFile::is_valid INSTRUMENT != 'EMERLIN'" << endl;
        return false;
    }

    return true;
}


void dsp::EmerlinFile::open_file(const char* filename) {
    // This is the header file
    FILE *fptr = fopen (filename, "r");
    if (!fptr)
        throw Error (FailedSys, "dsp::EmerlinFile::open_file",
                "fopen(%s) failed", filename);

    // Read the header
    char header[4096];
    fread(header, sizeof(char), 4096, fptr);
    fclose(fptr);

    // Get the data file
    if (ascii_header_get (header, "DATAFILE", "%s", datafile) < 0)
        throw Error (InvalidParam, "dsp::EmerlinFile::open_file",
                "Missing DATAFILE keyword");

    // Parse the standard ASCII info.  Timestamps are in VDIF packets
    // so not required.  Also we'll assume VDIF's "nchan" really gives
    // the number of polns for now, and NCHAN is 1.  NBIT is in VDIF packets.
    // We'll compute TSAMP from the bandwidth.  NDIM (real vs complex sampling)
    // is in VDIF packets via the iscomplex param.
    ASCIIObservation* info_tmp = new ASCIIObservation;
    info = info_tmp;

    info_tmp->set_required("UTC_START", false);
    info_tmp->set_required("OBS_OFFSET", false);
    info_tmp->set_required("NPOL",true);
    info_tmp->set_required("NBIT", false);
    info_tmp->set_required("NDIM", false);
    info_tmp->set_required("NCHAN", false);
    info_tmp->set_required("TSAMP", false);
    info_tmp->set_required("CALFREQ", false);
    info_tmp->load(header);




    // open the file
    fd = ::open (datafile, O_RDONLY);
    if (fd < 0)
        throw Error (FailedSys, "dsp::EmerlinFile::open_file()", 
                "open(%s) failed", filename);


  // Read until we get a valid frame
  bool got_valid_frame = false;
  char rawhdr_bytes[VDIF_HEADER_BYTES];
  vdif_header *rawhdr = (vdif_header *)rawhdr_bytes;
  int nbyte;
  while (!got_valid_frame)
  {
    size_t rv = read(fd, rawhdr_bytes, VDIF_HEADER_BYTES);
    if (rv != VDIF_HEADER_BYTES)
        throw Error (FailedSys, "EmerlinFile::open_file",
                "Error reading first header");

    // Get frame size
    nbyte = getVDIFFrameBytes(rawhdr);
    if (verbose) cerr << "EmerlinFile::open_file FrameBytes = " << nbyte << endl;
    //header_bytes = 0;
    //block_bytes = nbyte;
    //block_header_bytes = VDIF_HEADER_BYTES; // XXX what about "legacy" mode

    resolution=(nbyte-VDIF_HEADER_BYTES)*2*4*1; // in samples

    // If this first frame is invalid, go to the next one
    if (getVDIFFrameInvalid(rawhdr)==0)
      got_valid_frame = true;
    else
    {
      rv = lseek(fd, nbyte-VDIF_HEADER_BYTES, SEEK_CUR);
      if (rv<0)
        throw Error (FailedSys, "EmerlinFile::lseek",
            "Error seeking to next VDIF frame");
    }
  }

  // Rewind file
  lseek(fd, 0, SEEK_SET);
// Get basic params

  int nbit = getVDIFBitsPerSample(rawhdr);
  if (verbose) cerr << "EmerlinFile::open_file NBIT = " << nbit << endl;
  get_info()->set_nbit (nbit);

  bool iscomplex = rawhdr->iscomplex;
  if (iscomplex)
  {
    get_info()->set_ndim(2);
    get_info()->set_state(Signal::Analytic);
  }
  else
  {
    get_info()->set_ndim(1);
    get_info()->set_state(Signal::Nyquist);
  }
  if (verbose) cerr << "EmerlinFile::open_file iscomplex = " << iscomplex << endl;

  get_info()->set_nchan( 1 );
  get_info()->set_rate( (double) get_info()->get_bandwidth() * 1e6
      / (double) get_info()->get_nchan()
      * (get_info()->get_state() == Signal::Nyquist ? 2.0 : 1.0));
  if (verbose) cerr << "EmerlinFile::open_file rate = " << get_info()->get_rate() << endl;

  // Figure frames per sec from bw, pkt size, etc
  //double frames_per_sec = 64000.0;
  int frame_data_size = nbyte - VDIF_HEADER_BYTES;
  double frames_per_sec = get_info()->get_nbit() * get_info()->get_nchan() * get_info()->get_npol()
    * get_info()->get_rate() / 8.0 / (double) frame_data_size;
  if (verbose) cerr << "EmerlinFile::open_file frame_data_size = "
    << frame_data_size << endl;
  if (verbose) cerr << "EmerlinFile::open_file frames_per_sec = "
    << frames_per_sec << endl;

  // Set load resolution equal to one frame? XXX
  // This broke file unloading somehow ... wtf..
  //resolution = info.get_nsamples(frame_data_size);


  int mjd = getVDIFFrameMJD(rawhdr);
  int sec = getVDIFFrameSecond(rawhdr);
  int fn = getVDIFFrameNumber(rawhdr);
  first_second = getVDIFFullSecond(rawhdr);
  cur_frame=fn;
  if (verbose) cerr << "EmerlinFile::open_file MJD = " << mjd << endl;
  if (verbose) cerr << "EmerlinFile::open_file sec = " << sec << endl;
  if (verbose) cerr << "EmerlinFile::open_file fn  = " << fn << endl;
  get_info()->set_start_time( MJD(mjd,sec,(double)fn/frames_per_sec) );

  // Figures out how much data is in file based on header sizes, etc.
  set_total_samples();

    if (verbose)
        cerr << "EmerlinFile::open exit" << endl;
}





int64_t dsp::EmerlinFile::load_bytes(unsigned char* buffer, uint64_t nbytes) {

    int npol=get_info()->get_npol();
    int frame_length = 8000 * npol;

    if (nbytes % frame_length){
        // trim to an integer number of frames
        std::cerr << "dsp::EmerlinFile::load_bytes ERROR: Need to read integer number of frames" << std::endl;
        nbytes = frame_length*(nbytes/frame_length);
    }



    unsigned nframe = nbytes / frame_length;
    unsigned npacket = nframe/npol;

    std::memset(buffer, 0, nbytes); // zero the memory

    unsigned char* write_to = buffer;

    uint64_t to_load=nbytes;

    
    while (to_load > 0){

        char rawhdr_bytes[VDIF_HEADER_BYTES];
        vdif_header *rawhdr = (vdif_header *)rawhdr_bytes;

        size_t rv = read(fd, rawhdr_bytes, VDIF_HEADER_BYTES);
        if (rv != VDIF_HEADER_BYTES)
            throw Error (FailedSys, "EmerlinFile::load_bytes",
                    "Error reading header");

        // this is the full second this frame is relative to
        int64_t sec = getVDIFFullSecond(rawhdr);
        // this is the frame number in the second
        int64_t fn = getVDIFFrameNumber(rawhdr);
        // this is the stream number. For emerlin that means which polarisation.
        int64_t sn = getVDIFThreadID(rawhdr);

        fn += 4000*(sec-first_second);
        if (npol==1) {
            // in single polarisation mode we have files that could be either pol 0 or pol 1
            // In that case we want to ignore the stream number because all data comes from the same stream.
            // Otherwise we would have an offset of 1 frame in Pol 1 data in the byte_offset below
            sn = 0;
        }

        int byte_offset = ((fn-cur_frame)*npol + sn)*8000;

        //fprintf(stderr,"read %d/%d, pkt=%d to_load=%d %ld\n",fn,sn,byte_offset/8000,to_load,sec-first_second);
        if ((byte_offset+8000) > nbytes) {
            // we are past the requested data.
            // there is surely a better way than this!
            dropped += to_load/8000;
            std::cerr << "Some packets missing (left toload=" << to_load<<", total dropped so far = " << dropped << ")" << std::endl;
            fprintf(stderr,"read %d/%d, pkt=%d to_load=%d %ld\n",fn,sn,byte_offset/8000,to_load,sec-first_second);
            rv = lseek(fd, -VDIF_HEADER_BYTES, SEEK_CUR);
            if (rv<0)
                throw Error (FailedSys, "EmerlinFile::lseek",
                        "Error seeking to next VDIF frame");

            break;
        }

        write_to = buffer+byte_offset;


            rv = read(fd,write_to, 8000);

        if (rv!=8000){
            std::cerr << "dsp::EmerlinFile::load_bytes couldn't load data" << std::endl;
        }


/*        if(fn%2==sn){
            for(int i=0; i < 8000; ++i){
                write_to[i]=85;
            }
        } else {
        }*/

        to_load -= rv;
    }
    cur_frame += nframe;

    return nbytes;
}


int64_t dsp::EmerlinFile::seek_bytes(uint64_t bytes) {
    std::cerr << "dsp::EmerlinFile::seek_bytes NOT IMPLEMENTED "<<bytes << std::endl;
    return 0;
}

