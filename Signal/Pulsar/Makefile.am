
lib_LTLIBRARIES = libdsppsr.la

nobase_include_HEADERS = \
dsp/Archiver.h                  dsp/SubFold.h \
dsp/Fold.h                      dsp/TimeDivide.h \
dsp/LoadToFold.h                dsp/UnloaderShare.h \
dsp/LoadToFold1.h               dsp/PhaseLockedFilterbank.h \
dsp/LoadToFoldConfig.h          dsp/PhaseSeries.h \
dsp/LoadToFoldN.h               dsp/PhaseSeriesUnloader.h

libdsppsr_la_SOURCES = \
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

LDADD = libdsppsr.la \
	$(top_builddir)/More/General/libdspdsp.la \
	$(top_builddir)/Base/libdspbase.la \
	$(top_builddir)/Util/libdsputil.la

