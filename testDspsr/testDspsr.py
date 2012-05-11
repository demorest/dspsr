import os
import sys
import subprocess
import numpy as np
import cPickle
import time

def unpickle(fname):
    fh = open(fname,'r')
    res = cPickle.load(fh)
    fh.close()
    return res

def pickle(obj,fname,protocol=2):
    fh = open(fname,'w')
    cPickle.dump(obj,fh,protocol=protocol)
    fh.close()

inputFile = '/data/cyclo/guppi_55965_unknown_0001.0000.raw'
goldDir = '/data/cyclo/gold'
#commonArgs = "-r -D 1000 -E /home/gej/pulsar/parfiles/crab_t1.par -A -L 1 -T 10 -a PSRFITS -overlap"
dspsr = 'dspsr'

baseArgs = dict(r='', # Report timing
                D=1000, # dispersion measure
                E='/home/gej/pulsar/parfiles/crab_t1.par', # parfile
                A='', # all subints in one file
                L=1, # 1 second subints
                T=10, # total time processed
                a='PSRFITS', #outptu format
                overlap='', # needed for guppi/cyclic
                d=2, # ouput pols
                b=256, # number of bins
                e='fits', # no output extension
                
            )

def findClosestDict(d,dictList,ignoreKeys=[]):
    scores = []
    for n,dcomp in enumerate(dictList):
        score = 0
        if d == dcomp:
            print "found exact match"
            return n,len(d.keys()), dcomp
        for k in d.keys():
            if k not in ignoreKeys:
                try:
                    if d[k] == dcomp[k]:
                        print "found matching key:",k
                        score +=1
                    else:
                        print "key",k,"doesn't match",d[k],dcomp[k]
                        score -=1
                except KeyError:
                    score -= 1
#        for k in set(dcomp.keys()) - set(d.keys()):
#            if k not in ignoreKeys:
#                print "subtracting for missing key:",k
#                score -= 1
        scores.append(score)
    rank = np.array(scores).argsort()
    if (len(rank)>1) and (rank[-1] == rank[-2]):
        print "warning! found multiple with same score"
    return rank[-1], scores[rank[-1]], dictList[rank[-1]]
    
def findGoldStandard(args,ignoreKeys=[]):
    try:
        fdata = unpickle(os.path.join(goldDir,'index.pkl'))
    except:
        print "no gold standards yet"
        return None
    files = fdata.keys()
    argDicts = [fdata[f]['args'] for f in files]
    idx,score,best = findClosestDict(args,argDicts,ignoreKeys=ignoreKeys)
    if score < len(args.keys()):
        print "Did not find perfect match, best candidate was:"
        print files[idx]
        print "score:",score,"outof:",len(args.keys())
        return None
    else:
        return files[idx], fdata[files[idx]]
                    
def runDspsr(args,outdir,outpre):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    argList = [('-%s %s' % (k,v)) for (k,v) in args.items()]
    argList.insert(0,dspsr)
    outfile = outpre + '_' + time.strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(outdir,outfile)
    argList.append('-O %s' % outfile)
    argList.append(inputFile)
    logFile = (outfile + '.log')
    argList.append('1> %s 2> %s' % (logFile,logFile))
    cmd = ' '.join(argList)
    print "running: ", cmd
    tic = time.time()
    sys.stdout.flush()
    os.system(cmd)
    elapsed = time.time()-tic
    print "finished in %.2f minutes" % (elapsed/60)
    timingData = extractTiming(logFile)
    outfile = outfile + '.fits'
    info = dict(args=args, elapsed=elapsed, cmd = cmd, inputFile=inputFile, timestamp=time.time(),timingData=timingData)
    
    try:
        fdata = unpickle(os.path.join(outdir,'index.pkl'))
    except:
        print "creating new file index"
        fdata = {}
    fdata[outfile] = info
    print "writing file index"
    pickle(fdata, os.path.join(outdir,'index.pkl'))
    genIndexList(outdir)
    return outfile,info

def genIndexList(dirname):
    """
    Generate human readable index of dspsr runs from index.pkl file
    """
    outfile = os.path.join(dirname,'index.txt')
    fdata = unpickle(os.path.join(dirname,'index.pkl'))
    fnames = fdata.keys()
    fnames.sort()
    fh = open(outfile,'w')
    for fn in fnames:
        arglist = fdata[fn]['args'].keys()
        arglist.sort()
        fh.write("#### %s\n" % fn)
        fh.write(" %s  InputFile: %s\n Arguments:\n" % (time.ctime(fdata[fn]['timestamp']), fdata[fn]['inputFile']))
        for arg in arglist:
            fh.write("    %10s : %s\n" % (arg,fdata[fn]['args'][arg]))
        fh.write("\n %s\n  Timing data: \n" % fdata[fn]['cmd'])
        tdat = fdata[fn]['timingData']
        tlist = tdat.keys()
        tlist.sort()
        for targ in tlist:
            fh.write("   %20s: %.3f s\n" % (targ,tdat[targ]))
        if fdata[fn].has_key('stats'):
            fh.write("\n   Stats:\n")
            stats = fdata[fn]['stats']
            arglist = stats.keys()
            arglist.sort()
            for arg in arglist:
                fh.write("   %20s: %s\n" % (arg,stats[arg]))
                
        fh.write("\n Elapsed time: %.2f minutes \n\n" % (fdata[fn]['elapsed']/60.0))
    fh.close()

def extractTiming(logfile):
    fh = open(logfile,'r')
    fh.seek(0,2) # go to end of file
    fsize = fh.tell() # size of file
    if fsize > 10000:
        fh.seek(-10000,2) #assume data we want is in last 10000 bytes
    else:
        fh.seek(0)
    lines = fh.readlines()
    fh.close()
    start = [n for n,l in enumerate(lines) if l.startswith("Operation")]
    if not start:
        print "no timing data found for", logfile
        return {}
    results = {}
    ln = start[0]+1
    while True: 
        parts = lines[ln].split()
        if len(parts) != 3:
            break
        try:
            results[parts[0]] = float(parts[1])
        except:
            break
        ln += 1
    return results

def compareOutputs(refFile,testFile):
    import psrchive
    try:
        refdat = psrchive.Archive_load(refFile).get_data()
        testdat = psrchive.Archive_load(testFile).get_data()
    except Exception, e:
        print "Couldn't open files",e
        return dict(result="file not found")
    if refdat.shape != testdat.shape:
        print "Data are not the same shape, cannot compare!"
        print refFile,":",refdat.shape
        print testFile,":",testdat.shape
        return dict(result="incompatible")
    err = (refdat-testdat).flatten()
    abserr = np.abs(err)
    relerr = err/(refdat.flatten())
    inputrms = np.sqrt((refdat.flatten()**2).sum())
    res = dict(
        absmaxerr = abserr.max(),
        absmeanerr = abserr.mean(),
        relrms = np.sqrt((relerr**2).sum()),
        rmserr = np.sqrt((err**2).sum()),
        refFile = refFile,
        testFile = testFile
        )
    res['rmsrelerr'] = res['rmserr']/inputrms
    if res['rmsrelerr'] < 0.05:
        res['result'] = 'OK'
    else:
        res['result'] = 'wrong'
    return res
    
def updateInfo(fname,dirname,info):
    fdata = unpickle(os.path.join(dirname,'index.pkl'))
    fdata[fname] = info
    pickle(fdata, os.path.join(dirname,'index.pkl'))
    genIndexList(dirname)    

def cleanupIndex(dirname):
    fdata = unpickle(os.path.join(dirname,'index.pkl'))
    for k in fdata.keys():
        if not os.path.exists(k):
            print "did not find", k
            fdata.pop(k)
    pickle(fdata, os.path.join(dirname,'index.pkl'))
    genIndexList(dirname)

def testFilterbankCUDA(nchanList=[16], nbinList=[256], outpolsList=[2], redoRef=False,
                       refArgs = dict(V=''), cudaArgs = dict(V='',cuda=0), outdir = '/data/cyclo/testFbCuda'):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for outpols in outpolsList:
        for nbin in nbinList:
            for nchan in nchanList:
                print "nchan=%d, nbin=%d, npol=%d" % (nchan,nbin,outpols)
                localArgs = dict(b=nbin,F=('%d:D'%nchan),d=outpols)
                args = baseArgs.copy()
                args.update(refArgs)
                args.update(localArgs)
                
                refFileData = findGoldStandard(args)
                if refFileData is None:
                    print "Did not find reference file, creating"
                    refFilename, refInfo = runDspsr(args, goldDir, 'goldFB')
                else:
                    refFilename, refInfo = refFileData
                print "Reference file: ",refFilename
                timestr = time.strftime("%Y%m%d_%H%M%S")

                prefix = 'cudaFB_nchan%d_nbin%d_npol%d' % (nchan,nbin,outpols)
                print "running test"
                args = baseArgs.copy()
                args.update(cudaArgs)
                args.update(localArgs)
                testOutputFile,info = runDspsr(args,outdir,prefix)
                print "comparing output"
                res = compareOutputs(refFilename, testOutputFile)
                print res
                print "updating info"
                info['stats'] = res
                updateInfo(testOutputFile, outdir, info)
                
                
def testCyclicCUDA(nchanList=[16], cyclicList=[64], nbinList=[256], outpolsList=[2], redoRef=False,
                       refArgs = dict(V=''), cudaArgs = dict(V='',cuda=0), outdir = '/data/cyclo/testCyclicCuda'):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for cyclic in cyclicList:
        for outpols in outpolsList:
            for nbin in nbinList:
                for nchan in nchanList:
                    print "nchan=%d, nbin=%d, npol=%d, cyclic=%d" % (nchan,nbin,outpols,cyclic)
                    localArgs = dict(b=nbin,F=('%d:D'%nchan),d=outpols,cyclic=cyclic)
                    args = baseArgs.copy()
                    args.update(refArgs)
                    args.update(localArgs)
                    
                    refFileData = findGoldStandard(args)
                    if refFileData is None:
                        print "Did not find reference file, creating"
                        refFilename, refInfo = runDspsr(args, goldDir, 'goldCyclic')
                    else:
                        refFilename, refInfo = refFileData
                    print "Reference file: ",refFilename
                    timestr = time.strftime("%Y%m%d_%H%M%S")
    
                    prefix = 'cudaCyclic_nchan%d_nbin%d_npol%d_cyclic%d' % (nchan,nbin,outpols,cyclic)
                    print "running test"
                    args = baseArgs.copy()
                    args.update(cudaArgs)
                    args.update(localArgs)
                    testOutputFile,info = runDspsr(args,outdir,prefix)
                    print "comparing output"
                    res = compareOutputs(refFilename, testOutputFile)
                    print res
                    print "updating info"
                    info['stats'] = res
                    updateInfo(testOutputFile, outdir, info)                