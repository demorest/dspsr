
noinst_LTLIBRARIES = libmark5b.la

nobase_include_HEADERS = dsp/Mark5bFile.h dsp/Mark5bUnpacker.h

libmark5b_la_SOURCES = Mark5bFile.C Mark5bUnpacker.C

libmark5b_la_LIBADD = @MARK5ACCESS_LIBS@

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += @MARK5ACCESS_CFLAGS@
