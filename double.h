#include <thrust/future.h>
#include <thrust/host_vector.h>

thrust::device_event Double(thrust::host_vector<float>& doubles, thrust::host_vector<int>& ints);
