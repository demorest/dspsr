
#include "S2TwoBitCorrection.h"


dsp::S2TwoBitCorrection::S2TwoBitCorrection (unsigned nsample,
					     float cutoff_sigma)
{
}

void dsp::S2TwoBitCorrection::unpack ()
{
}


#if 0

s2_2bit_correct::PackType s2_2bit_correct::packtype = s2_2bit_correct::AT;
s2_2bit_correct::ResyncType s2_2bit_correct::resynctype = s2_2bit_correct::OFF;

int raw_2Bit_Value(int sign, int mag)
{
  int value = 0;

  if(mag==1 && sign==1)
    value = 3;
  else if(mag==0 && sign==1)
    value = 1;
  else if(mag==1 && sign==0)
    value = -1;
  else
    value = -3;
 
  return value;
}

void init_lookupAT (float * voltages) {
  unsigned int mask = 0x00000001;
  float sign, magnitude;
  int count = 0;

  for (unsigned int i=0;i<256;i++){
    sign = 1.0 - 2.0 * (mask & (i>>0));
    magnitude = 1.0 + 2.0 * (mask & (i>>4));
    voltages[count++] = sign * magnitude;
    sign = 1.0 - 2.0 * (mask & (i>>2));
    magnitude = 1.0 + 2.0 * (mask & (i>>6));
    voltages[count++] = sign * magnitude;
    sign = 1.0 - 2.0 * (mask & (i>>1));
    magnitude = 1.0 + 2.0 * (mask & (i>>5));
    voltages[count++] = sign * magnitude;
    sign = 1.0 - 2.0 * (mask & (i>>3));
    magnitude = 1.0 + 2.0 * (mask & (i>>7));
    voltages[count++] = sign * magnitude;
  }
}

// same as above but this code works for VLBA coding schemes not Normal AT
//
void init_lookupVLBA (float * voltages) {
  unsigned int mask = 0x00000001;
  int count = 0;

  for (unsigned int i=0;i<256;i++){
    voltages[count] = raw_2Bit_Value(mask & (i>>0), mask & (i>>4));
    count++;
    voltages[count] = raw_2Bit_Value(mask & (i>>2), mask & (i>>6));
    count++;
    voltages[count] = raw_2Bit_Value(mask & (i>>1), mask & (i>>5));
    count++;
    voltages[count] = raw_2Bit_Value(mask & (i>>3), mask & (i>>7));
    count++;
  }
}

//
s2_2bit_correct::s2_2bit_correct (int ppwt, float co_sigma)
{
  if (verbose) cerr << "s2_2bit_correct:: "
		    << " ppwt=" << ppwt
		    << " cutoff=" << co_sigma << "sigma\n";

  // this scheme is based on unpacking one complete unsigned int at a time
  assert (sizeof(uint16) == 2);

  set_twobit_limits (ppwt, co_sigma);

  // sets ppweight and cutoff_sigma calculates n_min and n_max
  maxstates = 256*4; // wonder lookup
  channels = 2;      // L,R

  // must initialize the following before calling size_dataspace();
  // ppweight, n_min, n_max, maxstates, channels

#ifdef _DEBUG
  cerr << "s2_2bit_correct:: call size_dataspace with"
       << "\n ppwt = " << ppwt
       << "\n chan = " << channels
       << "\n stat = " << maxstates
       << endl;
#endif

  size_dataspace ();

  // Generate 8-bit lookup table.
  float * lu8 = new float [maxstates];
  assert (lu8 != NULL);
  // Use the correct encoding scheme
  if(packtype == s2_2bit_correct::AT)
      init_lookupAT(lu8);
  else if(packtype == s2_2bit_correct::VLBA)
      init_lookupVLBA(lu8);
  else {
    string error("S2_2bit_correct: unknown packtype");
    cerr << error << endl;
    throw error;
  }

  // build the lookup tables
  float root_pi  = sqrt(M_PI);
  float root2    = sqrt(2.0);

  register float* dls_lut = dls_lookup;
  register float* spc_lut = spc_lookup;

  // Only generate tables for within cutoff_sigma of the mean.
  float min_a, max_a, min_b, max_b, min_A, max_A;
  min_a = min_b = min_A = MAXFLOAT;
  max_a = max_b = max_A = 0;

  for (int n_in=n_min; n_in <= n_max; n_in++)  {  
    /* Given n_in, the number of samples between x2 and x4, 
       then p_in is the left-hand side of Eq.44 */
    float p_in = (float) n_in / (float) ppwt;
    
    /* The inverse error function of p_in gives alpha, equal to the 
       "t/(root2 * sigma)" in brackets on the right-hand side of Eq.45 */
    float alpha = ierf (p_in);
    float expon = exp (-alpha*alpha);
    
    /* Equation 41 (ones: -y2, y3), substituting the above-computed values */
    float a = 2.0/(root2*alpha) * sqrt( 1.0 - (2.0*alpha/root_pi)*
					(expon/p_in));
    /* Similarly, Equation 40 (threes: -y1, y4) */
    float b = 2.0/(root2*alpha) * sqrt( 1.0 + (2.0*alpha/root_pi)*
					(expon/(1.0 - p_in)));
    
    /* And, finally, Equation 43... */
    float halfrootnum = a*(1-expon) + b*expon;
    float A = ( 2 * halfrootnum * halfrootnum ) /
      ( M_PI * ((a*a-b*b) * p_in + b*b) );
    *spc_lut = A;
    spc_lut ++;
    
    float* lu8ptr = lu8; 
    for (int sample=0; sample<maxstates; sample++) {
      float val = *lu8ptr;
      if ((val * val) == 9.0) val *= b/3.0; else val *= a;
      *dls_lut = val;
      lu8ptr++;  dls_lut++;
    }

    //fprintf (stderr, "%%ones:%f a:%f  b:%f\n", p_in, a, b);

    if (a > max_a)
      max_a = a;
    if (b > max_b)
      max_b = b;
    if (A > max_A)
      max_A = A;
    if (a < min_a)
      min_a = a;
     if (b < min_b)
      min_b = b;
    if (A < min_A)
      min_A = A;
  }

#ifdef _DEBUG
  fprintf (stderr, " a: min:%f max:%f \n b: min:%f max:%f \n A: min:%f max:%f\n",
	   min_a, max_a, min_b, max_b, min_A, max_A);
#endif

  assert (dls_lut - dls_lookup == maxstates * (n_max-n_min+1));

  delete [] lu8;
}

void s2_2bit_correct::unpack (float_Stream* fs, const Bit_Stream& bs)
{
  if (bs.nbit != 2) {
    string error ("s2_2bit_correct::unpack bit stream is not 2 bit sampled");
    cerr << error << endl;
    throw error;
  }

  if (fs->get_ppweight() != ppweight) {
    string error ("s2_2bit_correct::unpack float_Stream.ppweight != a2d_correct.ppweight");
    cerr << error << endl;
    throw error;
   }

  int n_in_lu[16]={4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0}; // # 0's in last 4 bits

  uint64 points_left=0;

  for (int ipol=0; ipol < bs.npol; ipol ++) {

    double start_seconds = fmod (fs->start_time.fracday()*86400.0, 
			       PKS_RESYNCH_PERIOD);

    double incr_seconds = fs->get_ppweight()/fs->rate;

    // For each ppweight pts, count the number of ones and
    // then decode using the relevant scaling factor.
    
    // rotate in the Magnitudes
    int rotatorm = 4 + 8 * ipol;
#if MACHINE_LITTLE_ENDIAN
    rotatorm = 12 - ipol * 8;
#endif

    // rotate in the whole 4 samples (sign and magnitudes)
    int rotator = 8 * ipol;
#if MACHINE_LITTLE_ENDIAN
    rotator = 8 - ipol * 8;
#endif
    
    register uint16 mask4=0x000F;
    register uint16 mask8=0x00FF;
    
    float * table_val;

    uint16* rawptr = (uint16*) bs.data;
    float* datptr = fs->data[ipol];

    uint16 offset;
    
    int samples_per_uint16 = sizeof(uint16) * 2;

    unsigned long* hist = histograms + ipol * ppweight;

    int* weights = fs->get_weights(ipol);

    // points in the float_Stream  ... should be checked against Bit_Stream?
    points_left = fs->ndat;

    for (int i=0;i<fs->get_nweights();i++){
      // Count n_in
      int n_in = 0;

      // //////////////////////////////////////////////////////////////////
      //
      // invalid assumption - Bit_Stream need not necessarily contain an
      //   integer multiple of ppweight points
      //
      int points = ppweight;
      if (points > points_left)
	points = points_left;

      int nwords = points/samples_per_uint16;
      points = nwords * samples_per_uint16;
      points_left -= points;

      if (packtype == s2_2bit_correct::AT) {
	for (int j=0;j<nwords;j++) {
	  n_in += n_in_lu [(*rawptr>>rotatorm)&mask4];
	  rawptr ++;
	}
      }
      else if (packtype == s2_2bit_correct::VLBA) {
	unsigned char oneval;
	for (int j=0;j<nwords;j++) {
	  // take the exclusive NOR of the sign and magnitude bits to
	  // count the number of ones
	  oneval = *rawptr>>rotator;
	  n_in += n_in_lu [(~(  oneval ^ (oneval>>4) )) & mask4];
	  rawptr ++;
	}
      }
      
      hist[n_in] ++;
      
      // rewind rawptr
      rawptr -= nwords;
      
      // Check for the bad data interval and flag if necessary.
      
      // If (MJD>so_and_so) check....
      if(resynctype == s2_2bit_correct::ON && packtype == s2_2bit_correct::AT)
	{
	  if (fs->telescope == TELID_PKS) {  // Parkes
	    if(start_seconds < PKS_RESYNCH_END) n_in = 0;
	    if(start_seconds+incr_seconds > PKS_RESYNCH_START) n_in = 0;
	  }
	  else if (fs->telescope == TELID_ATCA) {
	    if(start_seconds < ATCA_RESYNCH_END) n_in = 0;
	    if(start_seconds+incr_seconds > ATCA_RESYNCH_START) n_in = 0;
	  }
	}

      if (verbose && n_in == 0)
	cerr << "s2_2bit_correct::unpack Telescope Nulling" << endl;

      // If worse than 6 sigma, set to zero and weights likewise
      // Otherwise set weights to n_in and copy precomputed values.
      
      if ( (weights[i] == 0) ||
	   (n_in<n_min || n_in>n_max) ) {  // Bad data say so and set to zero. 
	
	for (int j=0; j<points; j++) {
	  *datptr = 0.0;
	  datptr ++;
	}
	rawptr += nwords;
	weights[i] = 0;

      }
      
      else {                               // data quality ok. Proceed
	
	weights[i]=n_in;
	// work out revelant section of the look-up table.
	float * fourvals = dls_lookup + (n_in-n_min) * maxstates;

	// Assign the values.
	for (int j=0; j<nwords; j++) {
	  offset = (int)((*rawptr>>rotator)&mask8);
	  rawptr++;
	  table_val = fourvals + 4 * offset;
	  *datptr++ = *table_val++;
	  *datptr++ = *table_val++;
	  *datptr++ = *table_val++;
	  *datptr++ = *table_val;

	}
      }
      
      start_seconds += (double)fs->get_ppweight()/fs->rate;
      start_seconds = fmod (start_seconds, PKS_RESYNCH_PERIOD);
    }

  }  // for each poln

  // this fix should truncate the float_Stream without any adverse effects
  if (points_left) {
    fs->ndat -= points_left;
    fs->end_time = fs->start_time + fs->ndat;
  }

  if (bs.npol == 2) {
    int* weights[2];
    weights[0] = fs->get_weights(0);
    weights[1] = fs->get_weights(1);

    for (unsigned iwt=0; iwt < fs->get_nweights(); iwt++) {
      if (weights[0][iwt] == 0 || weights[1][iwt] == 0)
	weights[0][iwt] = weights[1][iwt] = 0;
    }
  }
}

#endif
