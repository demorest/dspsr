
noinst_LTLIBRARIES = libgmrt.la

nobase_include_HEADERS = dsp/GMRTFile.h dsp/GMRTUnpacker.h \
	dsp/GMRTFilterbankFile.h dsp/GMRTFilterbank16.h

libgmrt_la_SOURCES = GMRTFile.C GMRTUnpacker.C data2rcv.h \
	gmrt.c gmrt.h gmrt_params.h

if HAVE_PRESTO
libgmrt_la_SOURCES += GMRTFilterbankFile.C GMRTFilterbank16.C
endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += @PRESTO_CFLAGS@

libgmrt_la_LIBADD = @PRESTO_LIBS@

