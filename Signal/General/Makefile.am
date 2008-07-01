
lib_LTLIBRARIES = libdspdsp.la

nobase_include_HEADERS = dsp/ACFilterbank.h dsp/Detection.h	       \
	dsp/ResponseProduct.h dsp/Accumulator.h dsp/SLDetect.h	       \
	dsp/Apodization.h dsp/SampleDelay.h dsp/AutoCorrelation.h      \
	dsp/SampleDelayFunction.h dsp/Example.h dsp/Shape.h	       \
	dsp/Bandpass.h dsp/Simultaneous.h dsp/Stats.h dsp/Filterbank.h \
	dsp/BitStatsPlotter.h dsp/Switcher.h dsp/Buffer.h	       \
	dsp/IncoherentFilterbank.h dsp/TScrunch.h dsp/LevelHistory.h   \
	dsp/TimeOrder.h dsp/Chomper.h dsp/LevelMonitor.h	       \
	dsp/TwoBitHistoryPlotter.h dsp/TwoBitStatsPlotter.h	       \
	dsp/Convolution.h dsp/Dedispersion.h dsp/DedispersionHistory.h \
	dsp/RFIFilter.h dsp/DedispersionSampleDelay.h dsp/Response.h   \
	dsp/Rescale.h dsp/PScrunch.h

libdspdsp_la_SOURCES = optimize_fft.c cross_detect.c cross_detect.h   \
	cross_detect.ic stokes_detect.c stokes_detect.h		      \
	stokes_detect.ic ACFilterbank.C TScrunch.C Accumulator.C      \
	TimeOrder.C Apodization.C AutoCorrelation.C Filterbank.C      \
	IncoherentFilterbank.C Bandpass.C LevelMonitor.C RFIFilter.C  \
	Chomper.C Response.C ResponseProduct.C Convolution.C	      \
	SLDetect.C Dedispersion.C SampleDelay.C DedispersionHistory.C \
	Shape.C DedispersionSampleDelay.C Detection.C Rescale.C \
	PScrunch.C

bin_PROGRAMS = dmsmear digitxt digimon

dmsmear_SOURCES = dmsmear.C 
digitxt_SOURCES = digitxt.C
digimon_SOURCES = digimon.C

if HAVE_PGPLOT

  ###########################################################################
  #
  # PGPLOT-specific code
  #

  libdspdsp_la_SOURCES += TwoBitStatsPlotter.C BitStatsPlotter.C
  bin_PROGRAMS += digistat passband

  digistat_SOURCES = digistat.C
  passband_SOURCES = passband.C

endif

if HAVE_sigproc
  bin_PROGRAMS += sigproc_filterbank
  sigproc_filterbank_SOURCES = sigproc_filterbank.C
endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

LDADD = libdspdsp.la $(top_builddir)/Kernel/libdspbase.la @PGPLOT_LIBS@

AM_CPPFLAGS += @PGPLOT_CFLAGS@

