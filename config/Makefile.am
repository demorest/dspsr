
#############################################################################
#
# This Makefile smooths over various bugs and bumps and other features
# that break the other GNU autotools-generated Makefiles
#

all: aclocal_update

MAKEFILE_ACLOCAL = $(top_srcdir)/config/Makefile.aclocal

include $(MAKEFILE_ACLOCAL)

$(MAKEFILE_ACLOCAL): $(PSRCHIVE_ACLOCAL)/Makefile.aclocal
	cp $(PSRCHIVE_ACLOCAL)/Makefile.aclocal $(MAKEFILE_ACLOCAL)

$(PSRCHIVE_ACLOCAL)/Makefile.aclocal:
	@echo; \
	echo A required configuration file is missing; \
	echo Please update your PSRCHIVE installation; \
	echo; \
	exit -1


