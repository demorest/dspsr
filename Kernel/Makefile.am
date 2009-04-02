
SUBDIRS = Classes Formats . Applications

lib_LTLIBRARIES = libdspbase.la
libdspbase_la_SOURCES  =
libdspbase_la_LIBADD = \
	Classes/libClasses.la \
	Formats/libFormats.la

LDFLAGS += $(PSRCHIVE_LIBS)
CPPFLAGS += $(PSRCHIVE_CPPFLAGS)

include $(top_srcdir)/config/Makefile.local

