
lib_LTLIBRARIES = libdspdsp.la

nobase_include_HEADERS = dsp/ACFilterbank.h dsp/Detection.h	       \
	dsp/ResponseProduct.h dsp/Accumulator.h	       \
	dsp/Apodization.h dsp/SampleDelay.h dsp/AutoCorrelation.h      \
	dsp/SampleDelayFunction.h dsp/Example.h dsp/Shape.h	       \
	dsp/Bandpass.h dsp/Simultaneous.h dsp/Stats.h dsp/Filterbank.h \
	dsp/BitStatsPlotter.h dsp/Switcher.h dsp/Buffer.h	       \
	dsp/IncoherentFilterbank.h dsp/TScrunch.h dsp/LevelHistory.h   \
	dsp/TimeOrder.h dsp/Chomper.h dsp/LevelMonitor.h	       \
	dsp/ExcisionHistoryPlotter.h dsp/ExcisionStatsPlotter.h	       \
	dsp/Convolution.h dsp/Dedispersion.h dsp/DedispersionHistory.h \
	dsp/RFIFilter.h dsp/DedispersionSampleDelay.h dsp/Response.h   \
	dsp/Rescale.h dsp/BandpassMonitor.h dsp/PScrunch.h	       \
	dsp/FourthMoment.h dsp/PolnCalibration.h dsp/Dump.h

libdspdsp_la_SOURCES = optimize_fft.c cross_detect.c cross_detect.h   \
	cross_detect.ic stokes_detect.c stokes_detect.h		      \
	stokes_detect.ic ACFilterbank.C TScrunch.C Accumulator.C      \
	TimeOrder.C Apodization.C AutoCorrelation.C Filterbank.C      \
	IncoherentFilterbank.C Bandpass.C LevelMonitor.C RFIFilter.C  \
	Chomper.C Response.C ResponseProduct.C Convolution.C	      \
	Dedispersion.C SampleDelay.C DedispersionHistory.C \
	Shape.C DedispersionSampleDelay.C Detection.C Rescale.C	      \
	PScrunch.C BandpassMonitor.C FourthMoment.C Stats.C \
	PolnCalibration.C Dump.C

if HAVE_CUFFT

nobase_include_HEADERS += dsp/FilterbankCUDA.h dsp/TransferCUDA.h dsp/DetectCUDA.h

libdspdsp_la_SOURCES += FilterbankCUDA.cu TransferCUDA.C DetectCUDA.cu 

libdspdsp_la_LIBADD = FilterbankCUDA.lo DetectCUDA.lo

FilterbankCUDA.lo : FilterbankCUDA.cu
	@CUDA_NVCC@ -c $(srcdir)/FilterbankCUDA.cu
	sed -e 's/Filterbank/FilterbankCUDA/' Filterbank.lo > FilterbankCUDA.lo

DetectCUDA.lo : DetectCUDA.cu
	@CUDA_NVCC@ -c $(srcdir)/DetectCUDA.cu
	sed -e 's/Detection/DetectCUDA/' Detection.lo > DetectCUDA.lo
endif

bin_PROGRAMS = dmsmear digitxt digimon digihist

dmsmear_SOURCES = dmsmear.C 
digitxt_SOURCES = digitxt.C
digimon_SOURCES = digimon.C
digihist_SOURCES = digihist.C

check_PROGRAMS = test_PolnCalibration test_OptimalFFT

test_PolnCalibration_SOURCES = test_PolnCalibration.C
test_OptimalFFT_SOURCES = test_OptimalFFT.C

if HAVE_PGPLOT

  ###########################################################################
  #
  # start PGPLOT-specific code
  #

  libdspdsp_la_SOURCES += ExcisionStatsPlotter.C BitStatsPlotter.C
  bin_PROGRAMS += digistat passband

  digistat_SOURCES = digistat.C
  passband_SOURCES = passband.C

  #
  # end PGPLOT-specific code
  #
  ###########################################################################

endif


if HAVE_sigproc

  ###########################################################################
  #
  # start sigproc-specific code
  #

  bin_PROGRAMS += digifil
  digifil_SOURCES = digifil.C

if HAVE_dada
  bin_PROGRAMS += the_decimator
  the_decimator_SOURCES = the_decimator.C
  the_decimator_LDADD = $(LDADD) @OPENSSL_LIBS@ @PSRXML_LIBS@
  the_decimator_CPPFLAGS = $(AM_CPPFLAGS) $(CPPFLAGS) @PSRXML_CFLAGS@
endif

  #
  # end sigproc-specific code
  #
  ###########################################################################

endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

LDADD = libdspdsp.la $(top_builddir)/Kernel/libdspbase.la \
	@PGPLOT_LIBS@ @CUFFT_LIBS@ @CUDA_LIBS@

AM_CPPFLAGS += @PGPLOT_CFLAGS@ @CUFFT_CFLAGS@
AM_CXXFLAGS = @OPENMP_CFLAGS@


