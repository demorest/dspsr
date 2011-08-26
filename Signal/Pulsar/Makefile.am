
lib_LTLIBRARIES = libdspsr.la

nobase_include_HEADERS = \
dsp/Archiver.h                  dsp/Subint.h \
dsp/Fold.h                      dsp/TimeDivide.h \
dsp/UnloaderShare.h \
dsp/LoadToFold1.h               dsp/PhaseLockedFilterbank.h \
dsp/LoadToFoldConfig.h          dsp/PhaseSeries.h \
dsp/LoadToFoldN.h               dsp/PhaseSeriesUnloader.h \
dsp/CyclicFold.h

libdspsr_la_SOURCES = \
Archiver.C                            \
ArchiverExtensions.C    TimeDivide.C            \
Fold.C                  UnloaderShare.C \
LoadToFold1.C           PhaseLockedFilterbank.C dsp_verbosity.C \
LoadToFoldConfig.C      PhaseSeries.C  \
LoadToFoldN.C           PhaseSeriesUnloader.C \
CyclicFold.C

if HAVE_CUFFT

nobase_include_HEADERS += dsp/FoldCUDA.h dsp/TransferPhaseSeriesCUDA.h

libdspsr_la_SOURCES += FoldCUDA.cu TransferPhaseSeriesCUDA.C

libdspsr_la_LIBADD = FoldCUDA.lo

FoldCUDA.lo : FoldCUDA.cu
	@CUDA_NVCC@ -c $(srcdir)/FoldCUDA.cu
	sed -e 's/Fold/FoldCUDA/' Fold.lo > FoldCUDA.lo

endif

bin_PROGRAMS = dspsr

dspsr_SOURCES = dspsr.C 

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

LDADD = libdspsr.la \
	$(top_builddir)/Signal/General/libdspdsp.la \
	$(top_builddir)/Kernel/libdspbase.la \
	$(top_builddir)/Signal/Statistics/libdspstats.la \
	@CUFFT_LIBS@ @CUDA_LIBS@

AM_CPPFLAGS += @CUFFT_CFLAGS@
AM_CXXFLAGS = @OPENMP_CFLAGS@

