
/* multiplies two complex arrays (ntrans number of complex points) */
int mfilter (const unsigned ntrans, float* spectrum, float* filter)
{
  unsigned i;
  float* d_r = spectrum;
  float* d_i = spectrum + 1;
  float* f_r = filter;
  float* f_i = filter + 1;

  float newd_r;

  for (i=0;i<ntrans;i++) {
    newd_r = *f_r * *d_r - *f_i * *d_i;
    *d_i = *f_i * *d_r + *f_r * *d_i;
    *d_r = newd_r;
    d_r+=2; d_i+=2; f_r+=2; f_i+=2;
  }

  return 0;
}

