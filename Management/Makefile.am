
bin_SCRIPTS = dspsr_ldflags dspsr_cflags
noinst_SCRIPTS = release.csh

all:
	-chmod +x $(noinst_SCRIPTS)

libs: fix_prefix

install-exec-hook: fix_prefix

#
# The following lines ensure that the dspsr_cflags and dspsr_ldflags
# scripts are up-to-date with respect to where the software was installed.
#

FIX_PREFIX = ${prefix}/bin/dspsr_ldflags ${prefix}/bin/dspsr_cflags

fix_prefix: install-binSCRIPTS
	perl -p -i -e "s|PREFIX|${prefix}|" ${FIX_PREFIX}

