#pragma once
#include <ATen/Parallel.h>
#include <stdexcept>
namespace lane {
inline void configure_cpu_threads(int threads) {
    if (threads < 1) throw std::runtime_error("--threads doit etre >= 1");
    at::set_num_threads(threads); // CPU intra-op; a initialiser avant l'entrainement/inference
}
}
