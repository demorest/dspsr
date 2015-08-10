
lib_LTLIBRARIES = libdspdsp.la

nobase_include_HEADERS = dsp/ACFilterbank.h dsp/Detection.h	       \
	dsp/ResponseProduct.h dsp/Accumulator.h			       \
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
	dsp/FourthMoment.h dsp/PolnCalibration.h dsp/Dump.h	       \
	dsp/OptimalFFT.h dsp/OptimalFilterbank.h dsp/on_host.h	       \
	dsp/FScrunch.h dsp/FilterbankBench.h dsp/FilterbankConfig.h    \
	dsp/FilterbankEngine.h dsp/filterbank_engine.h		       \
	dsp/GeometricDelay.h					       \
	dsp/TFPFilterbank.h dsp/RFIZapper.h dsp/SKFilterbank.h	       \
	dsp/Resize.h dsp/SKDetector.h dsp/SKMasker.h		       \
	dsp/Pipeline.h dsp/SingleThread.h dsp/MultiThread.h            \
	dsp/PolnSelect.h

libdspdsp_la_SOURCES = optimize_fft.c cross_detect.c cross_detect.h  \
	cross_detect.ic stokes_detect.c stokes_detect.h		     \
	stokes_detect.ic ACFilterbank.C TScrunch.C      \
	TimeOrder.C Apodization.C AutoCorrelation.C Filterbank.C     \
	IncoherentFilterbank.C Bandpass.C LevelMonitor.C RFIFilter.C \
	Chomper.C Response.C ResponseProduct.C Convolution.C	     \
	Dedispersion.C SampleDelay.C DedispersionHistory.C	     \
	Shape.C DedispersionSampleDelay.C Detection.C Rescale.C	     \
	PScrunch.C BandpassMonitor.C FourthMoment.C Stats.C	     \
	PolnCalibration.C Dump.C OptimalFFT.C FScrunch.C	     \
	FilterbankBench.C OptimalFilterbank.C FilterbankConfig.C \
	GeometricDelay.C mfilter.c \
	TFPFilterbank.C RFIZapper.C SKFilterbank.C \
	Resize.C SKDetector.C SKMasker.C \
	SingleThread.C MultiThread.C dsp_verbosity.C \
	PolnSelect.C

bin_PROGRAMS = dmsmear digitxt digimon digihist filterbank_speed

if HAVE_CUFFT

nobase_include_HEADERS += CUFFTError.h dsp/LaunchConfig.h \
	dsp/FilterbankCUDA.h dsp/filterbank_cuda.h \
	dsp/TransferCUDA.h \
	dsp/TransferBitSeriesCUDA.h dsp/SKMaskerCUDA.h dsp/DetectionCUDA.h \
	dsp/ConvolutionCUDA.h

libdspdsp_la_SOURCES += CUFFTError.C LaunchConfig.C FilterbankCUDA.cu TransferCUDA.C \
	TransferBitSeriesCUDA.C DetectionCUDA.cu SKMaskerCUDA.cu ConvolutionCUDA.cu

if HAVE_CUFFT_CALLBACKS
bin_PROGRAMS += cufft_callback_bench

# device relocated object file linked to device code
ConvolutionCUDA_DC.o: ConvolutionCUDA.o
	$(CUDA_NVCC) -o ConvolutionCUDA_DC.o -dlink ConvolutionCUDA.o -lcufft_static

cufft_callback_bench_DC.o: cufft_callback_bench.o
	$(CUDA_NVCC) -o cufft_callback_bench_DC.o -dlink cufft_callback_bench.o -lcufft_static

ConvolutionCUDA.lo: ConvolutionCUDA.cu
	$(top_srcdir)/config/cudalt.py $(top_builddir)/libtool $@ $(CUDA_NVCC) -dc -c $<
	$(CUDA_NVCC) -o ConvolutionCUDA_DC.o -dlink ConvolutionCUDA.o -lcufft_static

cufft_callback_bench.lo: cufft_callback_bench.cu
	$(top_srcdir)/config/cudalt.py $(top_builddir)/libtool $@ $(CUDA_NVCC) -dc -c $<
	$(CUDA_NVCC) -o cufft_callback_bench_DC.o -dlink cufft_callback_bench.o -lcufft_static

cufft_callback_bench_LDADD = $(LDADD) cufft_callback_bench_DC.o -lcudart -lcufft_static -lculibos

endif
endif


dmsmear_SOURCES = dmsmear.C 
digitxt_SOURCES = digitxt.C
digimon_SOURCES = digimon.C
digihist_SOURCES = digihist.C
filterbank_speed_SOURCES = filterbank_speed.C

check_PROGRAMS = test_PolnCalibration test_OptimalFFT

test_PolnCalibration_SOURCES = test_PolnCalibration.C
test_OptimalFFT_SOURCES = test_OptimalFFT.C

if HAVE_PGPLOT

  ###########################################################################
  #
  # start PGPLOT-specific code
  #

  libdspdsp_la_SOURCES += ExcisionStatsPlotter.C BitStatsPlotter.C
  libdspdsp_la_LIBADD = @PGPLOT_LIBS@
  bin_PROGRAMS += digistat passband

  digistat_SOURCES = digistat.C
  digistat_LDADD = @PSRPLOT_LIBS@ $(LDADD)

  passband_SOURCES = passband.C
  passband_LDADD = @PSRPLOT_LIBS@ $(LDADD)

if HAVE_CUFFT_CALLBACKS
  digistat_LDADD += ConvolutionCUDA_DC.o -lcudart -lcufft_static -lculibos
  passband_LDADD += ConvolutionCUDA_DC.o -lcudart -lcufft_static -lculibos
endif
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

  nobase_include_HEADERS += dsp/LoadToFil.h dsp/LoadToFilN.h
  libdspdsp_la_SOURCES += LoadToFil.C LoadToFilN.C

  bin_PROGRAMS += digifil
  digifil_SOURCES = digifil.C


if HAVE_dada
  bin_PROGRAMS += the_decimator
  the_decimator_SOURCES = the_decimator.C
  the_decimator_LDADD = $(LDADD) @OPENSSL_LIBS@ @PSRXML_LIBS@ @PSRDADA_LIBS@
  the_decimator_CPPFLAGS = $(AM_CPPFLAGS) $(CPPFLAGS) @PSRXML_CFLAGS@ @PSRDADA_CFLAGS@

if HAVE_CUFFT_CALLBACKS
  the_decimator_LDADD += ConvolutionCUDA_DC.o -lcudart -lcufft_static -lculibos
endif

endif

  #
  # end sigproc-specific code
  #
  ###########################################################################

endif

#if HAVE_CUFFT_CALLBACKS
#  digimon_LDADD = $(LDADD) ConvolutionCUDA_DC.o -lcudart -lcufft_static -lculibos
#  digifil_LDADD = $(LDADD) ConvolutionCUDA_DC.o -lcudart -lcufft_static -lculibos
#  digihist_LDADD = $(LDADD) ConvolutionCUDA_DC.o -lcudart -lcufft_static -lculibos
#endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

LDADD = libdspdsp.la \
	$(top_builddir)/Kernel/libdspbase.la \
	$(top_builddir)/Signal/Statistics/libdspstats.la \
	@PGPLOT_LIBS@ @CUDA_LIBS@ @CUFFT_LIBS@

AM_CPPFLAGS += @PGPLOT_CFLAGS@ @CUFFT_CFLAGS@ 
AM_CXXFLAGS = @OPENMP_CFLAGS@

