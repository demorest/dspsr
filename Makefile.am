
SUBDIRS	= config Kernel Signal Management

include $(top_srcdir)/config/Makefile.root

ACLOCAL_AMFLAGS = -I config

EXTRA_DIST += config/backends.default config/formats.sh config/Makefile.aclocal

DISTCLEANFILES = backends.list

backends.list:
	cp $(top_srcdir)/config/backends.default backends.list

