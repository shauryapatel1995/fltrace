/*
 * prefetch.c - Prefetcher backend implementation
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <pthread.h>
#include <dlfcn.h>
#include "rmem/common.h"
#include "rmem/prefetch.h"
#include "rmem/page.h"
#include "rmem/fault.h"
#include <sys/time.h>                

/*
 * Implementation to do the actual prefetching with predictions from a
 * prefetching policy. The function does the following - 
 * 1. Constructs the required input for a prefetching policy.
 * 2. Obtains predictions from a given policy.
 * 3. Converts predictions to addresses to prefetch
 * 4. Checks whether the page for an address can be prefetched.
 * 5. Prefetches the page, and updates the eviction count to reflect 
 * the additional pages.
*/
unsigned long page_postfetch(fault_t * f, FeatureVector *features, 
                                int *responses, int chan_id, int *nevicts_needed)
{
#ifdef benchmark_model
    struct timeval t1, t2;
    double elapsed_time;
#endif
    void *local_addr = bkend_buf_alloc();
    pgthread_t owner_kthr;
    pgflags_t oldflags;
    int nprefetches = 0, noverflow = 0;
    enum fault_status status;
    unsigned long long pressure;
    uint64_t *ptr = (uint64_t *) f->page;
    int faulting_location = (f->faulting_addr - f->page) / sizeof(uint64_t);

    assert(ptr);
    /* Setup pointer features */
    for(int i = 0; i < 512; i++, ptr++) {
        FeatureVector *feature = &features[i];
        feature->pc = f->pc;
        feature->offset = i;
        feature->delta = (*ptr - f->page) / 4096;
        feature->offset_from_faulting = i - faulting_location;
        responses[i] = 0;
    }
    /* Setup next-N features */
    for (int i = 512; i < 600; i++) {
        FeatureVector *feature = &features[i];
        feature->pc = f->pc;
        feature->offset = 0; 
        feature->delta = i - 511; 
        feature->offset_from_faulting = 0;
        responses[i] = 0;
    } 
#ifdef benchmark_model
    gettimeofday(&t1, NULL);
#endif
    /* This part of the code runs inference on all 600 
     * candidates. For now we are changing the strategy to
     * run if the ptr is actually present.
     *  page_postfetch_preds(features, responses);
     */
#ifdef benchmark_model
    gettimeofday(&t2, NULL);
	elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000000;      // sec to ms
    elapsed_time += (t2.tv_usec - t1.tv_usec);   // us to ms
    printf("%f us.\n", elapsed_time);
#endif
    /* 
     * 1. Generate address value.
     * 2. Do the same checks as the ones in readahead plus walking
     * the page table.
     * If is_page_prefetchable succeeds, the page is locked.
     * Unlocking needs to be managed by this function.
     * 3. Run inference on page.
     * 4. Call local post read on the address after making a fault?
     * Or decide on a design for local post read.
     * 5. Call fault_read_done for the page.
     * 6. clear the pages after fetching is done.
     */
    for (int i = 0; i < 512; i++) { 
        assert(f);
        uint64_t ptr_val = *((uint64_t *) f->page + i); 
        ptr_val = ptr_val & ~CHUNK_MASK;
        if (is_page_prefetchable(f, ptr_val)) {
            /* Do the inference to get prediction */
            page_postfetch_preds(&features[i], responses, 1);
            if (responses[0] == 1 && ptr_val != 0) {
                // fprintf(stdout, "Pointer prefetch address: %lu\n", ptr_val);
                /* Copy the page into the local buffer from remote */
                if(rmbackend->post_read_prefetch(chan_id, f, ptr_val, local_addr)) {
                    clear_page_flags(f->mr, ptr_val, PFLAG_WORK_ONGOING, &oldflags);
                    goto out; 
                }
                /* Complete prefetch read by copying the page into memory */
                prefetch_read_done(ptr_val, local_addr, f);
                RSTAT(PREFETCHES)++;
                nprefetches++;
                /* Free the page */
                clear_page_flags(f->mr, ptr_val, PFLAG_WORK_ONGOING, &oldflags);
                
                /* check that the page was locked and that the saved kthread thread id
                * (if this fault was originally from a kthread) is same or different 
                * the from current kthread id depending on if the fault was stolen */
                assert(!!(oldflags & PFLAG_WORK_ONGOING));
                if (owner_kthr) {
                    assert(f->stolen_from_cq || owner_kthr == current_kthread_id);
                    assert(!f->stolen_from_cq || owner_kthr != current_kthread_id);
                }
            } else {
                clear_page_flags(f->mr, ptr_val, PFLAG_WORK_ONGOING, &oldflags);
            }   
        }
    }

    for (int i = 512; i < 516; i++) {
        uint64_t ptr_val =  f->page + (i - 511); 
        ptr_val = ptr_val & ~CHUNK_MASK;
        if(is_page_prefetchable(f, ptr_val)) { 
            // fprintf(stdout, "Prefetch address: %lu\n", ptr_val);
            page_postfetch_preds(&features[i], responses, 1);
            if (responses[0] == 1) {
                /* Copy the page into a local buffer from remote */
                if(rmbackend->post_read_prefetch(chan_id, f, ptr_val, local_addr)) {
                    //XXX(shaurp): Remove the pgflag ongoing flag here
                    goto out; 
                }
                prefetch_read_done(ptr_val, local_addr, f);
                RSTAT(PREFETCHES)++;
                fprintf(stderr, "Prefetch\n");
                nprefetches++;
                clear_page_flags(f->mr, ptr_val, 
                    PFLAG_WORK_ONGOING, &oldflags);
                /* check that the page was locked and that the saved kthread thread id
                * (if this fault was originally from a kthread) is same or different 
                * the from current kthread id depending on if the fault was stolen */
                assert(!!(oldflags & PFLAG_WORK_ONGOING));
                if (owner_kthr) {
                    assert(f->stolen_from_cq || owner_kthr == current_kthread_id);
                    assert(!f->stolen_from_cq || owner_kthr != current_kthread_id);
                }
            } else {
                clear_page_flags(f->mr, ptr_val, 
                    PFLAG_WORK_ONGOING, &oldflags);
            }
        }
    }
   
    /*
     * 5. Unmap the pages from the kernel, mark the page as prefetched
     * in our data structures (Complete after the entire workflow is done).
     * 6. Calculate/update nevict for the prefetched pages. 
     */ 
    assert(nevicts_needed);
    /* book some memory for the pages */
    if (nprefetches > 0) {
        assert(nprefetches > 0);
        pressure = atomic64_add_and_fetch(&memory_used, nprefetches * CHUNK_SIZE);
        log_debug("%s - memory pressure during prefetch %llu, limit %lu", FSTR(f), 
            pressure, local_memory);
        if (pressure > local_memory) {
            noverflow = (pressure - local_memory) / CHUNK_SIZE;
            *nevicts_needed = (noverflow < nprefetches) ? noverflow : nprefetches;
        }

        /* update maximum memory usage counter. FIXME: should use CAS! */
        if (pressure > atomic64_read(&max_memory_used))
            atomic64_write(&max_memory_used, pressure);

        log_debug("%s - %d page(s) prefetched with return status %d, pressure %llu"
            " evicts %d", FSTR(f), nprefetches, status, pressure, *nevicts_needed);
    }
out: 
    bkend_buf_free(local_addr);
    return 0; 
}
