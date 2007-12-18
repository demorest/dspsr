
bin_SCRIPTS = dspsr_ldflags

libs: fix_prefix

install-exec-hook: fix_prefix

#
# The following lines ensure that the dspsr_cflags and dspsr_ldflags
# scripts are up-to-date with respect to where the software was installed.
#

FIX_PREFIX := ${prefix}/bin/dspsr_ldflags

LOCAL_ROOT := ${top_srcdir}
include ${top_srcdir}/config/Makefile.backends

fix_prefix: install-binSCRIPTS
	perl -p -i -e "s|PREFIX|${prefix}|" ${FIX_PREFIX}
	perl -p -i -e "s|BACKENDS|${LBACKENDS}|" ${FIX_PREFIX}

