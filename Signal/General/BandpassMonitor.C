#include "dsp/BandpassMonitor.h"


dsp::BandpassMonitor::BandpassMonitor(){
	file[0] = fopen("bp0","w");
	file[1] = fopen("bp1","w");

	mean_sum[0] = NULL;
	mean_sum[1] = NULL;

	count[0] = 0;
	count[1] = 0;


}

void dsp::BandpassMonitor::append(uint64 start, uint64 end, int pol, int nchan, float* means, float* variances, float* freq){
	if(mean_sum[pol] == NULL){
		mean_sum[pol] = new float[nchan];
		var_sum[pol] = new float[nchan];
		for(int i = 0; i < nchan; i++){
			mean_sum[pol][i] = 0;
			var_sum[pol][i] = 0;
		}

	}


	fprintf(file[pol],"#START#\n#%lld %lld\n",start,end);
	for(int i = 0; i < nchan; i++){
		fprintf(file[pol],"%f %f %f\n",freq[i],means[i],variances[i]);
		mean_sum[pol][i] += means[i];
		var_sum[pol][i] += variances[i];
	}
	
	count[pol]++;

	char* str = new char[80];


	sprintf(str,"bpm%d",pol);

	FILE *fptr = fopen(str,"w");
	fprintf(fptr,"#START#\n#0 %lld\n",end);

	for(int i = 0; i < nchan; i++){
		fprintf(fptr,"%f %f %f\n",freq[i],mean_sum[pol][i]/(float)count[pol],var_sum[pol][i]/(float)count[pol]);
	}
	fclose(fptr);

	delete[] str;
}
