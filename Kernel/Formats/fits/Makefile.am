noinst_LTLIBRARIES = libfits.la

nobase_include_HEADERS = dsp/FITSUnpacker.h dsp/FITSFile.h \
			 dsp/GUPPIFITSUnpacker.h dsp/FITSDigitizer.h \
             dsp/FITSOutputFile.h

libfits_la_SOURCES = FITSUnpacker.C GUPPIFITSUnpacker.C \
		     FITSFile.C fits_params.h FITSDigitizer.C \
             FITSOutputFile.C

#############################################################################
#

libfits_la_LIBADD = @CFITSIO_LIBS@ -lgomp

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += @CFITSIO_CFLAGS@ -fopenmp

