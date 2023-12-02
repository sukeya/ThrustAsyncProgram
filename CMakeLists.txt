cmake_minimum_required(VERSION 3.27.7)

# CUDAアーキテクチャを指定する。
# 最初に設定しないと、プロジェクトが作成できません。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(my_program VERSION 0.1.0 LANGUAGES CXX CUDA)

# ここの設定はお好みで。
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OPTIMIZE_DEPENDENCIES ON)

# CUDAを見つける。
find_package(CUDAToolkit 12 REQUIRED)

# Thrustを見つける。
add_subdirectory(thrust)
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

# CUDAライブラリを作る。
add_library(double double.cu)
target_compile_options(
  double
  PRIVATE
    # ラムダ式をデバイスコードで使えるようにする。
    -expt-extended-lambda
    # SIMDを無効にする。
    # SIMDの無効化は必要ありませんが、Eigenなど線形代数ライブラリを使う際に必要になります。
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fno-tree-vectorize>"
    "$<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fno-vectorize>"
)
# ここはお好みで。
target_compile_features(
  double
  PRIVATE
    cuda_std_20
)
# 誰もインストールしないはずなので、簡略に。
target_include_directories(
  double
  PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
# CUDAランタイムとThrustをリンクする。
# 当たり前ですが、ThrustにはCUDAはないので。
target_link_libraries(
  double
  PUBLIC
    CUDA::cudart
    Thrust
)

# my_programを作る。
add_executable(my_program main.cpp)
target_link_libraries(
  my_program
  PRIVATE
    double
)
# ここはお好みで。
target_compile_features(
  my_program
  PRIVATE
    cxx_std_20
)
# ここはお好みで。
target_compile_options(
  my_program
  PRIVATE
    "$<$<CONFIG:Debug>:-g;-O0;-Wall;-Wextra>"
    "$<$<CONFIG:Release>:-DNDEBUG;-O3;-march=native;-mtune=native>"
)
# CUDA 5.0以前のバージョンでは、デバイスコードで使用する関数の実装までインクルードする必要が
# ありましたが、CUDA 5.0以降では、デバイスコードで使用する関数の宣言のみをインクルードするだけで
# 十分になりました。
# このことをコンパイラとリンカーに伝えるために、CUDA_SEPARABLE_COMPILATIONプロパティをONに
# しています。
# ただし、このプロパティは実行ファイルにのみONに設定して下さい。
set_property(TARGET my_program PROPERTY CUDA_SEPARABLE_COMPILATION ON)
