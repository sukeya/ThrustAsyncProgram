#include <thrust/async/copy.h>
#include <thrust/async/transform.h>
#include <thrust/device_vector.h>

#include "double.h"

template <class T>
thrust::device_event Double(thrust::host_vector<T>& ts) {
  // デバイス側の配列を用意
  auto device_ts = thrust::device_vector<T>();

  // メモリの確保と初期化
  device_ts.resize(ts.size());

  // ホストからデバイスへの非同期コピー
  auto copy_ts_event = thrust::async::copy(
      thrust::host,
      thrust::device,
      ts.begin(),
      ts.end(),
      device_ts.begin()
  );

  // デバイス側での計算
  auto double_ts_event = thrust::async::transform(
      thrust::device.after(copy_ts_event),
      device_ts.begin(),
      device_ts.end(),
      device_ts.begin(),
      [] __device__(T d) { return d * d; }
  );

  // デバイスからホストへの非同期コピー
  auto copy_back_ts_event = thrust::async::copy(
      thrust::device.after(double_ts_event),
      thrust::host,
      device_ts.begin(),
      device_ts.end(),
      ts.begin()
  );
  
  return copy_back_ts_event;
}

thrust::device_event Double(thrust::host_vector<float>& floats, thrust::host_vector<int>& ints) {
  // 非同期実行の完了をまとめる
  return thrust::when_all(Double(floats), Double(ints));
}
