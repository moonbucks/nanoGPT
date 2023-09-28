import sys

print ('parsing batch', sys.argv[1])
batch_id=sys.argv[1]

from hta.trace_analysis  import TraceAnalysis
analyzer = TraceAnalysis(trace_dir = f"./traces/batch_{batch_id}")

time_spent_df = analyzer.get_temporal_breakdown()
print(time_spent_df)
print()

### Idle time breakdown
#idle_time_df = analyzer.get_idle_time_breakdown()
#print(idle_time_df)
#print()
##
### Kernel breakdown
##kernel_breakdown_df = analyzer.get_gpu_kernel_breakdown()
##print(kernel_breakdown_df)
#print()
##
### Communication computation overlap
#comm_comp_overlap_df = analyzer.get_comm_comp_overlap()
#print(comm_comp_overlap_df)
#print()
#
#cuda_kernel_launch_stats = analyzer.get_cuda_kernel_launch_stats()
#print(cuda_kernel_launch_stats)
