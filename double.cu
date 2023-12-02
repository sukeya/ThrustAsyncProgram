#include <thrust/async/copy.h>
#include <thrust/async/transform.h>
#include <thrust/device_vector.h>

#include "double.h"

thrust::device_event Double(thrust::host_vector<float>& floats, thrust::host_vector<int>& ints) {
  auto device_floats = thrust::device_vector<float>();
  auto device_ints   = thrust::device_vector<int>();

  device_floats.resize(floats.size());
  device_ints.resize(ints.size());

  auto copy_ints_event = thrust::async::copy(
      thrust::host,
      thrust::device,
      ints.begin(),
      ints.end(),
      device_ints.begin()
  );
  auto copy_doubles_event = thrust::async::copy(
      thrust::host,
      thrust::device,
      floats.begin(),
      floats.end(),
      device_floats.begin()
  );
  auto double_ints_event = thrust::async::transform(
      thrust::device.after(copy_ints_event),
      device_ints.begin(),
      device_ints.end(),
      device_ints.begin(),
      [] __device__(int i) { return i * i; }
  );
  auto double_doubles_event = thrust::async::transform(
      thrust::device.after(copy_doubles_event),
      device_floats.begin(),
      device_floats.end(),
      device_floats.begin(),
      [] __device__(float d) { return d * d; }
  );
  auto copy_back_ints_event = thrust::async::copy(
      thrust::device.after(double_ints_event),
      device_ints.begin(),
      device_ints.end(),
      ints.begin()
  );
  auto copy_back_doubles_event = thrust::async::copy(
      thrust::device.after(double_doubles_event),
      device_floats.begin(),
      device_floats.end(),
      floats.begin()
  );

  return thrust::when_all(copy_back_ints_event, copy_back_doubles_event);
}