#! /bin/csh

set server = `git config --get remote.origin.url`

if ( $server =~ git://* ) then
  echo 'ERROR: this command can be run only by a developer'
  exit -1
endif

set SFUSER = `echo $server | awk -F// '{print $2}' | awk -F@ '{print $1}'`
set SFPATH = ${SFUSER},dspsr@web.sourceforge.net:htdocs/classes

set SFHTML = dspsr

echo "Installing $SFHTML as $SFUSER"

cd html

rsync -avz $SFHTML $SFPATH/

