
lib_LTLIBRARIES = libdspsr.la

nobase_include_HEADERS = \
dsp/Archiver.h                  dsp/SubFold.h \
dsp/Fold.h                      dsp/TimeDivide.h \
dsp/LoadToFold.h                dsp/UnloaderShare.h \
dsp/LoadToFold1.h               dsp/PhaseLockedFilterbank.h \
dsp/LoadToFoldConfig.h          dsp/PhaseSeries.h \
dsp/LoadToFoldN.h               dsp/PhaseSeriesUnloader.h

libdspsr_la_SOURCES = \
Archiver.C              SubFold.C               \
ArchiverExtensions.C    TimeDivide.C            \
Fold.C                  UnloaderShare.C \
LoadToFold1.C           PhaseLockedFilterbank.C dsp_verbosity.C \
LoadToFoldConfig.C      PhaseSeries.C  \
LoadToFoldN.C           PhaseSeriesUnloader.C

bin_PROGRAMS = dspsr

dspsr_SOURCES = dspsr.C 

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

LDADD = libdspsr.la \
	$(top_builddir)/Signal/General/libdspdsp.la \
	$(top_builddir)/Kernel/libdspbase.la

AM_CXXFLAGS = @OPENMP_CFLAGS@

