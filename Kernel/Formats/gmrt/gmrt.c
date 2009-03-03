
/*
  modified version of GMRT_hdr_to_inf from PRESTO by Scott Ransom.
*/

#include "dsp/infodata.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int need_byteswap_st = 0;
static int sb_flag;
static int bitshift = 2;
static char badch[100];

int GMRT_hdr_to_inf(char *datfilenm, infodata * idata)
/* Convert GMRT header into an infodata structure */
{
   FILE *hdrfile;
   double ss;
   char line[200], ctmp[100], project[20], date[20];
   char *hdrfilenm;
   int numantennas, hh, mm, cliplen;

   hdrfilenm = calloc(strlen(datfilenm) + 1, 1);
   cliplen = strlen(datfilenm) - 3;
   strncpy(hdrfilenm, datfilenm, cliplen);
   strcpy(hdrfilenm + cliplen, "hdr");
   hdrfile = fopen(hdrfilenm, "r");
   if (!hdrfile)
     return -1;

   while (fgets(line, 200, hdrfile)) {
      if (line[0] == '#') {
         continue;
      } else if (strncmp(line, "Site            ", 16) == 0) {
         sscanf(line, "%*[^:]: %[^\n]\n", idata->telescope);
      } else if (strncmp(line, "Observer        ", 16) == 0) {
         sscanf(line, "%*[^:]: %[^\n]\n", idata->observer);
      } else if (strncmp(line, "Proposal        ", 16) == 0) {
         sscanf(line, "%*[^:]: %[^\n]\n", project);
      } else if (strncmp(line, "Array Mode      ", 16) == 0) {
         sscanf(line, "%*[^:]: %[^\n]\n", idata->instrument);
      } else if (strncmp(line, "Observing Mode  ", 16) == 0) {
         continue;
      } else if (strncmp(line, "Date            ", 16) == 0) {
         sscanf(line, "%*[^:]: %[^\n]\n", date);
      } else if (strncmp(line, "Num Antennas    ", 16) == 0) {
         sscanf(line, "%*[^:]: %d\n", &numantennas);
      } else if (strncmp(line, "Antenna List    ", 16) == 0) {
         continue;
      } else if (strncmp(line, "Num Channels    ", 16) == 0) {
         sscanf(line, "%*[^:]: %d\n", &idata->num_chan);
      } else if (strncmp(line, "Channel width   ", 16) == 0) {
         sscanf(line, "%*[^:]: %lf\n", &idata->chan_wid);
      } else if (strncmp(line, "Frequency Ch.1  ", 16) == 0) {
         sscanf(line, "%*[^:]: %lf\n", &idata->freq);
      } else if (strncmp(line, "Sampling Time   ", 16) == 0) {
         sscanf(line, "%*[^:]: %lf\n", &idata->dt);
         idata->dt /= 1000000.0;        /* Convert from us to s */
      } else if (strncmp(line, "Num bits/sample ", 16) == 0) {
         sscanf(line, "%*[^:]: %d\n", &idata->num_bit);
      } else if (strncmp(line, "Data Format     ", 16) == 0) {
         {
            /* The following is from question 20.9 of the comp.lang.c FAQ */
            int x = 1, machine_is_little_endian = 0;
            if(*(char *)&x == 1) {
               machine_is_little_endian = 1;
            }

            if (strstr(line, "little")) {
               if (machine_is_little_endian)
                  need_byteswap_st = 0;
               else
                  need_byteswap_st = 1;
            } else {
               if (machine_is_little_endian)
                  need_byteswap_st = 1;
               else
                  need_byteswap_st = 0;
            }
         }
      } else if (strncmp(line, "Polarizations   ", 16) == 0) {
         sscanf(line, "%*[^:]: %[^\n]\n", ctmp);
         if (strcmp(ctmp, "Total I") == 0)
           idata->num_poln=1;
         else
          idata->num_poln=4;
      } else if (strncmp(line, "MJD             ", 16) == 0) {
         sscanf(line, "%*[^:]: %d\n", &idata->mjd_i);
      } else if (strncmp(line, "UTC             ", 16) == 0) {
         sscanf(line, "%*[^:]: %d:%d:%lf\n", &hh, &mm, &ss);
         idata->mjd_f = (hh + (mm + ss / 60.0) / 60.0) / 24.0;
      } else if (strncmp(line, "Source          ", 16) == 0) {
         sscanf(line, "%*[^:]: %[^\n]\n", idata->object);
      } else if (strncmp(line, "Coordinates     ", 16) == 0) {
         sscanf(line, "%*[^:]: %d:%d:%lf, %d:%d:%lf\n",
                &idata->ra_h, &idata->ra_m, &idata->ra_s,
                &idata->dec_d, &idata->dec_m, &idata->dec_s);
      } else if (strncmp(line, "Coordinate Sys  ", 16) == 0) {
         sscanf(line, "%*[^:]: %[^\n]\n", ctmp);
         if (strcmp(ctmp, "J2000")) {
            printf("\nWarning:  Cannot non-J2000 coordinates!\n"
                   "   Data is '%s'\n", ctmp);
         }
      } else if (strncmp(line, "Drift Rate      ", 16) == 0) {
         continue;
      } else if (strncmp(line, "Obs. Length     ", 16) == 0) {
         continue;
      } else if (strncmp(line, "Bad Channels    ", 16) == 0) {
         sscanf(line, "%*[^:]: %[^\n]\n", ctmp);
      } else if (strncmp(line, "Bit shift value ", 16) == 0) {
         sscanf(line, "%*[^:]: %d\n", &bitshift);
      } else {
         continue;
      }
   }
   idata->freqband = idata->chan_wid * idata->num_chan;
   idata->dm = 0.0;
   idata->N = 0.0;
   idata->numonoff = 0;
   idata->bary = 0;
   idata->fov = 0.0;

   strcpy(idata->band, "Radio");
   strcpy(idata->analyzer, "dspsr");
   strcpy(badch, ctmp);
   printf("bad channels are %s\n", badch);
   printf("bit shift value set to %d\n", bitshift);
   printf("No of bits/sample is %d\n", idata->num_bit);

   sprintf(idata->notes, "%d antenna observation for Project %s\n"
           "    UT Date at file start = %s\n"
           "    Bad channels: %s\n", numantennas, project, date, ctmp);
   fclose(hdrfile);
   free(hdrfilenm);

   return 0;
}


