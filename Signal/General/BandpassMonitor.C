#include "dsp/BandpassMonitor.h"

#include "Error.h" 

using namespace std;


dsp::BandpassMonitor::BandpassMonitor(){
	//file[0] = fopen("bp0","w");
	//file[1] = fopen("bp1","w");

	//file[2] = fopen("time0","w");
	//file[3] = fopen("time1","w");

	//file[4] = fopen("bp0rms","w");
	//file[5] = fopen("bp1rms","w");

// open in binary write mode
	file[0] = fopen("bp0.dat","wb");
	file[1] = fopen("bp1.dat","wb");

	file[2] = fopen("time0.dat","wb");
	file[3] = fopen("time1.dat","wb");

	file[4] = fopen("bp0rms.dat","wb");
	file[5] = fopen("bp1rms.dat","wb");

	mean_sum[0] = NULL;
	mean_sum[1] = NULL;

	count[0] = 0;
	count[1] = 0;


}

void dsp::BandpassMonitor::append(uint64 start, uint64 end, int pol, int nchan, float* means, float* variances, float* rmss, float* freq, float* zerotime){
	if(mean_sum[pol] == NULL){
		mean_sum[pol] = new float[nchan];
		var_sum[pol] = new float[nchan];
		for(int i = 0; i < nchan; i++){
			mean_sum[pol][i] = 0;
			var_sum[pol][i] = 0;
		}

	}


	//fprintf(file[pol],"#START#\n#%lld %lld\n",start,end);
	for(int i = 0; i < nchan; i++){
		//fprintf(file[pol],"%f %f %f\n",freq[i],means[i],variances[i]);
		//fprintf(file[pol],"%f %f \n",freq[i],means[i]);
		//fprintf(file[pol+4],"%f %f \n",freq[i],rmss[i]);
		mean_sum[pol][i] += means[i];
		var_sum[pol][i] += variances[i];
	}

	//binary write of bandpass and zerodm time series
	fwrite(means, 1, nchan*sizeof(float), file[pol]);
	fwrite(zerotime, 1, (end-start)*sizeof(float), file[pol+2]);
	fwrite(rmss, 1, nchan*sizeof(float), file[pol+4]);

	// dummy debug lines
	//fprintf(stderr,"end %d start %d diff %d \n",end,start,end-start);
	cerr << "Done " << end << " samples";
	cerr << "   \r";
	//fprintf(stderr,"end %d start %d diff %d \n",end,start,end-start);
	//fprintf(stderr,"size of zerotime %d size of float %d \n",sizeof(zerotime), sizeof(float));

	//ascii print of zero dm time series
	//for(int i = 0; i < (end-start); i++)
		//fprintf(file[pol+2],"%d %f \n",i,zerotime[i]);
	
	count[pol]++;

	char* str = new char[80];


	//sprintf(str,"bpm%d",pol);
	sprintf(str,"bpm%d.dat",pol);

	//FILE *fptr = fopen(str,"w");
	FILE *fptr = fopen(str,"wb");
	//fprintf(fptr,"#START#\n#0 %lld\n",end);

	for(int i = 0; i < nchan; i++){
		//fprintf(fptr,"%f %f %f\n",freq[i],mean_sum[pol][i]/(float)count[pol],var_sum[pol][i]/(float)count[pol]);
		means[i] = mean_sum[pol][i]/(float)count[pol];
	}
	fwrite(means, 1, nchan*sizeof(float), fptr);
	fclose(fptr);

	//close all files
	//for (pol = 0; pol < 6; pol++) fclose(file[pol]);

	delete[] str;
}
