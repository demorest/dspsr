/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "pumadata.h"

#include <stdio.h>
#include <string.h>

/* pretty much copied from pumahdr.c */

void pumadump (const Header_type *hdr, FILE* fptr, Boolean verb)
{
  int   i;
  char  label[40], label2[10],label3[9];

  fprintf(fptr,"\n");
  fprintf(fptr,"========================================");
  fprintf(fptr,"========================================\n");
  fprintf(fptr,"\n");
  fprintf(fptr,"File %27s which is file number %3d out of total %3d files\n",
              hdr->gen.ThisFileName, hdr->gen.FileNum, hdr->gen.NFiles);
  fprintf(fptr,"on tape %24s which is tape number %3d out of total %3d tapes\n",
              hdr->gen.TapeID, hdr->gen.TapeNum, hdr->gen.NTapes);
  fprintf(fptr,"\n");
  fprintf(fptr,  "GENERAL     Scannumber                 : %s\n",hdr->gen.ScanNum);
  fprintf(fptr,  "            Data file version          : %s\n",hdr->gen.HdrVer);
  fprintf(fptr,  "            Platform                   : %s\n",hdr->gen.Platform);
  fprintf(fptr,  "            Comment by operator        : %s\n",hdr->gen.Comment);
  fprintf(fptr,"\n");
  fprintf(fptr,  "            Data in this file starts   : %12.6f\n",hdr->gen.DataMJD*1.0 + 
                                                               hdr->gen.DataTime);
  fprintf(fptr,  "            Data in file from cluster  : ");
  for (i=0; i< MAXFREQBANDS; i++) {
    if (hdr->gen.Cluster[i] == TRUE)
      fprintf(fptr,"%1d",i);
    else
      fprintf(fptr,".");
  }
  fprintf(fptr,"\n");
  fprintf(fptr,  "            out of the active clusters : ");
  for (i=0; i< MAXFREQBANDS; i++) {
    if (hdr->mode.ActCluster[i] == TRUE)
      fprintf(fptr,"%1d",i);
    else
      fprintf(fptr,".");
  }
  fprintf(fptr,"\n");
  fprintf(fptr,"\n");
  fprintf(fptr,  "            NumBytes in header         : %12d\n", sizeof(hdr));
  fprintf(fptr,  "            NumBytes in par. block     : %12d\n", hdr->gen.ParBlkSize);
  fprintf(fptr,  "            NumBytes in data block     : %12d\n", hdr->gen.DataBlkSize);
  fprintf(fptr,  "            NumBytes total             : %12d\n", sizeof(hdr) + 
                                               hdr->gen.ParBlkSize + hdr->gen.DataBlkSize);
  fprintf(fptr,"\n");

  if (verb) {
    fprintf(fptr,"OBSERVATORY Name                       : %s\n",hdr->obsy.Name);
    fprintf(fptr,"            (west) Longitude           : %f rad\n",hdr->obsy.Long);
    fprintf(fptr,"            (north) Latitude           : %f rad\n",hdr->obsy.Lat);
    fprintf(fptr,"            Height                     : %f m\n",hdr->obsy.Height);
    fprintf(fptr,"\n");
  }

  fprintf(fptr,  "OBSERVATION ");
  if (verb) {
    fprintf(fptr,            "Proposal                   : %s\n",hdr->obs.ObsName);
    fprintf(fptr,"            Type                       : %s\n",hdr->obs.ObsType);
    fprintf(fptr,"            ScanNumber of last cal.obs : %s\n",hdr->obs.LastCalID);
    fprintf(fptr,"            ");
  }
  fprintf(fptr,              "Start date of observation  : %6d (MJD)\n",hdr->obs.StMJD);
  fprintf(fptr,  "            Start time (sec after 0h)  : %6d s\n",hdr->obs.StTime);
  if (verb) {
    fprintf(fptr,"            Start Local Siderial Time  : %f\n",hdr->obs.StLST);
  }
  fprintf(fptr,  "            Averaged maser offset      : %f s\n",hdr->obs.MaserOffset);
  fprintf(fptr,  "            Observation duration       : %15.8f s\n",hdr->obs.Dur);
  if (verb) {
    fprintf(fptr,"            Ionospheric RM correction  : %f\n",hdr->obs.IonRM);
  }
  fprintf(fptr,"\n");

  fprintf(fptr,  "TARGET      PulsarName                 : %s\n",hdr->src.Pulsar);
  if (verb) {
    fprintf(fptr,"            Ra                         : %f rad\n",hdr->src.RA);
    fprintf(fptr,"            Dec                        : %f rad\n",hdr->src.Dec);
    fprintf(fptr,"            Epoch                      : %s\n",hdr->src.Epoch);
  }
  fprintf(fptr,"\n");

  if (verb) {
    fprintf(fptr,"WSRT        Backend                    : %s\n",hdr->WSRT.Backend);
    fprintf(fptr,"            Residual Fringe Stopping   : ");
    if (hdr->WSRT.ResFringeOff == TRUE) 
      fprintf(fptr,"OFF (ok)\n");
    else
      fprintf(fptr,"ON  (might be a problem)\n");
    fprintf(fptr,"            Adding Box                 : ");                    
    if (hdr->WSRT.AddBoxOn == TRUE)       
      fprintf(fptr,"On (ok)\n");
    else
      fprintf(fptr,"OFF (problem)\n");
    fprintf(fptr,"\n");
  }

  fprintf(fptr,"MODE        Mode number                : %d\n",hdr->mode.Nr);
  fprintf(fptr,"            FIR Factor                 : %d\n",hdr->mode.FIRFactor);
  fprintf(fptr,"            Num Samples added wi SHARC : %d\n",hdr->mode.NSampsAdded);
  fprintf(fptr,"            Num SHARCs added           : %d\n",hdr->mode.NSHARCsAdded);
  fprintf(fptr,"            Num Freq Chan in this file : %d\n",hdr->mode.NFreqInFile);
  fprintf(fptr,"            Sampling Time              : %d ns\n",hdr->mode.Tsamp);
  fprintf(fptr,"            Output                     : ");
  if (hdr->mode.Iout == TRUE)
    fprintf(fptr,"I ");
  else 
    fprintf(fptr,". ");
  if (hdr->mode.Qout == TRUE)
    fprintf(fptr,"Q ");
  else
    fprintf(fptr,". ");
  if (hdr->mode.Uout == TRUE)
    fprintf(fptr,"U ");
  else
    fprintf(fptr,". ");
  if (hdr->mode.Vout == TRUE)
    fprintf(fptr,"V ");
  else
    fprintf(fptr,". ");
  if (hdr->mode.Xout == TRUE)
    fprintf(fptr,"X ");
  else
    fprintf(fptr,". ");
  if (hdr->mode.Yout == TRUE)
    fprintf(fptr,"Y ");
  else
    fprintf(fptr,". ");
  fprintf(fptr,"\n");
  fprintf(fptr,"            Num Dedispersed TimeSeries : %d",hdr->mode.NDMs);
  if (hdr->mode.NDMs != 0) {
    fprintf(fptr," -->");
    for (i=0; i < hdr->mode.NDMs; i++) {
      fprintf(fptr," %f",hdr->mode.DM[i]);
    }
  }
  fprintf(fptr,"\n");
  fprintf(fptr,"            Rotation Measure           : %f\n",hdr->mode.RM);
  fprintf(fptr,"            Is DC dynamically removed? : ");
  if (hdr->mode.DC_Dynamic == TRUE)
    fprintf(fptr,"YES\n");
  else
    fprintf(fptr,"NO\n");
  fprintf(fptr,"            Is Scale dynam. adjusted?  : ");
  if (hdr->mode.ScaleDynamic == TRUE)
    fprintf(fptr,"YES\n");
  else
    fprintf(fptr,"NO\n");
  fprintf(fptr,"            Adjustment interval        : %f\n",hdr->mode.AdjustInterval);
  fprintf(fptr,"            Output in                  : ");
  if (hdr->mode.FloatsOut == TRUE)
    fprintf(fptr,"Floats\n");
  else
    fprintf(fptr,"%d Bits\n",hdr->mode.BitsPerSamp);
  fprintf(fptr,"\n");

  fprintf(fptr,  "TELESCOPES  Telescope      0   1   2   3   4   5   6   7   8   9   A   B   C   D\n");
  fprintf(fptr,  "            Active      ");
  for (i=0; i< MAXTELESCOPES; i++) {
    if (hdr->WSRT.Tel[i].Active == TRUE)
      fprintf(fptr,"   +");
    else
      fprintf(fptr,"   -");
  }
  fprintf(fptr,"\n");
  fprintf(fptr,  "            NoiseSrcOff ");
  for (i=0; i< MAXTELESCOPES; i++) {
    if (hdr->WSRT.Tel[i].NoiseSrcOff == TRUE)
      fprintf(fptr,"   +");
    else
      fprintf(fptr,"   -");
  }
  fprintf(fptr,"\n");
  if (verb) {
    fprintf(fptr,"            FrontendID  ");
    for (i=0; i< MAXTELESCOPES; i++) {
      fprintf(fptr," %3.3s",hdr->WSRT.Tel[i].FrontendID);
    }
    fprintf(fptr,"\n");
    fprintf(fptr,"            Status      ");
    for (i=0; i< MAXTELESCOPES; i++) {
      fprintf(fptr," %3.3s",hdr->WSRT.Tel[i].FrontendStatus);
    }
    fprintf(fptr,"\n");
    fprintf(fptr,"            T_sys       ");
    for (i=0; i< MAXTELESCOPES; i++) {
      fprintf(fptr," %3.3d",(int) hdr->WSRT.Tel[i].Tsys);
    }
    fprintf(fptr,"\n");
  }
  fprintf(fptr,"\n");

  fprintf(fptr,  "BANDS       Band              0      1      2      3      4      5      6      7\n");
  fprintf(fptr,  "            Operational ");
  for (i=0; i< MAXFREQBANDS; i++) {
    if (hdr->WSRT.Band[i].Operational == TRUE)
      fprintf(fptr,"      +");
    else
      fprintf(fptr,"      -");
  }
  fprintf(fptr,"\n");
  fprintf(fptr,"            NonFlip     ");
  for (i=0; i< MAXFREQBANDS; i++) {
    if (hdr->WSRT.Band[i].NonFlip == TRUE)
      fprintf(fptr,"      +");
    else
      fprintf(fptr,"      -");
  }
  fprintf(fptr,"\n");
  fprintf(fptr,  "            MidFreq     ");
  for (i=0; i< MAXFREQBANDS; i++) {
    fprintf(fptr," %6.3f",hdr->WSRT.Band[i].MidFreq); 
  }
  fprintf(fptr,"\n");
  fprintf(fptr,  "            Width       ");
  for (i=0; i< MAXFREQBANDS; i++) {
    fprintf(fptr," %6.3f",hdr->WSRT.Band[i].Width); 
  }
  fprintf(fptr,"\n");
  fprintf(fptr,  "            SkyMidFreq  ");
  for (i=0; i< MAXFREQBANDS; i++) {
    fprintf(fptr," %6.1f",hdr->WSRT.Band[i].SkyMidFreq); 
  }
  fprintf(fptr,"\n");
  fprintf(fptr,  "            ConnToClustr");
  for (i=0; i< MAXFREQBANDS; i++) {
    fprintf(fptr," %6d",hdr->WSRT.BandsToClusMap[i]);
  }
  fprintf(fptr,"\n");
  fprintf(fptr,  "            XPolScaleFac");
  for (i=0; i< MAXFREQBANDS; i++) {
    fprintf(fptr," %6.4f",hdr->mode.XPolScaleFac[i]);
  }
  fprintf(fptr,"\n");

  fprintf(fptr,"\n");

  if (verb) {
    fprintf(fptr,"RCS Codes   Master : %s\n",hdr->software.Master);
    fprintf(fptr,"            DSP    : %s\n",hdr->software.DSP[0]);
    fprintf(fptr,"            Filter : %s\n",hdr->software.Filter);
    fprintf(fptr,"\n");
    fprintf(fptr,"EXITCODE    Exit Code WSRT             : %d\n",hdr->check.ExitCodeObsy);
    fprintf(fptr,"            Exit Code PuMa             : %d\n",hdr->check.ExitCodePuMa);
    fprintf(fptr,"            Exit Code Clusters         :");
    for (i=0; i<MAXFREQBANDS; i++) {
      fprintf(fptr," %d",hdr->check.ExitCodeClust[i]);
    }
    fprintf(fptr,"\n");
    fprintf(fptr,"            Exit Code after data check : %d\n",hdr->check.ExitCodeDataConsistency);
    fprintf(fptr,"\n");
  }

  if (verb) {
    fprintf(fptr,"\n");
    strcpy(label,"Unknown");
    strcpy(label2,"samples   ");
    strcpy(label3,"StartTime");
    if (hdr->redn.Raw == TRUE)
      strcpy(label,"Raw data");
    if (hdr->redn.IsDedisp == TRUE)
      strcpy(label,"Dedispersed data");
    if (hdr->redn.IsPwr == TRUE) {
      strcpy(label,"Power spectrum");
      strcpy(label2,"time dumps");
    }
    if (hdr->redn.Folded == TRUE) {
      strcpy(label2,"time dumps");
      strcpy(label3,"MidPoint ");
      if (hdr->redn.IsDedisp == TRUE) {
        strcpy(label,"Dedispersed folded data");
      }
      else
        strcpy(label,"Undedispersed folded data");
      }
    fprintf(fptr,"REDUCTION   Type                       : %s\n",label);
    fprintf(fptr,"            %9s of reduced data  : %19.13f\n",label3,
                                            hdr->redn.MJDint + hdr->redn.MJDfrac);
    fprintf(fptr,"            Number of %10s       : %d\n",label2,hdr->redn.NTimeInts);
    fprintf(fptr,"            Delta T between %10s : %f\n",label2,hdr->redn.DeltaTime);   
    fprintf(fptr,"            Number of frequency chans  : %d\n",hdr->redn.NFreqs);
    fprintf(fptr,"            Delta freq between chans   : %f\n",hdr->redn.DeltaFreq);   
    fprintf(fptr,"            Central frequency          : %f\n",hdr->redn.FreqCent);
    fprintf(fptr,"            Number of fold/pwr bins    : %d\n",hdr->redn.NBins);
    fprintf(fptr,"            Dispersion Measure used    : %f\n",hdr->redn.DM);
    fprintf(fptr,"            Zapfile used               : %s\n",hdr->redn.Zapfile);
    fprintf(fptr,"            Foldperiod used            : %.12f\n",hdr->redn.FoldPeriod);
    fprintf(fptr,"            Type of folding            : ");
    if (hdr->redn.Polyco == TRUE)
      fprintf(fptr,"polyco driven ");
    if (hdr->redn.Bary == TRUE)
      fprintf(fptr,"barycentered");
    fprintf(fptr,"\n");
    fprintf(fptr,"            Number of polyco coeff     : %d\n",hdr->redn.NCoef);
    fprintf(fptr,"            Time span of polyco        : %d\n",hdr->redn.PolycoSpan);
    fprintf(fptr,"            Size of FFT (coh. dedisp)  : %d\n",hdr->redn.CohFFTSize);
    fprintf(fptr,"            Time of reduction          : %29s",ctime((time_t*)&hdr->redn.TRedn));
    fprintf(fptr,"            Output                     : ");
    if (hdr->redn.OI == TRUE)
      fprintf(fptr,"I ");
    else
      fprintf(fptr,". ");
    if (hdr->redn.OQ == TRUE)
      fprintf(fptr,"Q ");
    else  
      fprintf(fptr,". ");
    if (hdr->redn.OU == TRUE)
      fprintf(fptr,"U ");
    else
      fprintf(fptr,". ");
    if (hdr->redn.OV == TRUE)
      fprintf(fptr,"V ");
    else
      fprintf(fptr,". ");
    if (hdr->redn.OX == TRUE)
      fprintf(fptr,"X ");
    else
      fprintf(fptr,". ");
    if (hdr->redn.OY == TRUE)
      fprintf(fptr,"Y ");
    else
      fprintf(fptr,". ");
    if (hdr->redn.OP == TRUE)
      fprintf(fptr,"P ");
    else  
      fprintf(fptr,". ");
    if (hdr->redn.OTheta == TRUE)
      fprintf(fptr,"theta ");
    else
      fprintf(fptr,". ");
    if (hdr->redn.Op == TRUE)
      fprintf(fptr,"p ");
    else
      fprintf(fptr,". ");
    if (hdr->redn.Ov == TRUE)
      fprintf(fptr,"v ");
    else
      fprintf(fptr,". ");
    if (hdr->redn.Opoldeg == TRUE)
      fprintf(fptr,"poldeg ");
    else
      fprintf(fptr,". ");

    fprintf(fptr,"\n");
    fprintf(fptr,"            Reduction command line     : %s\n",hdr->redn.Command);

  }  

  fprintf(fptr,"========================================");
  fprintf(fptr,"========================================\n");
  fprintf(fptr,"\n");
}
