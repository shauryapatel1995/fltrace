#ifndef __PREFETCH_H__
#define __PREFETCH_H__

/* Data structures */
typedef struct {
    uint32_t pc;
    int offset;
    float delta;
    int offset_from_faulting;
} FeatureVector;

typedef struct {
    int cache_hit;
    int cache_miss; 
    int misprefetch;
    double precision;
    double recall;
    double accuracy;
} PredictionMetrics;

/* methods */
void init_prefetcher();
unsigned long page_prefetch_preds(FeatureVector features[], int *response_arr);
unsigned long page_postfetch_preds(FeatureVector features[], int *response_arr);
unsigned long page_postfetch(fault_t *f, FeatureVector features[], int *response_arr);

#endif // __PREFETCH_H
