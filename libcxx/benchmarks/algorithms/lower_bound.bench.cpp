//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

#include "common.h"


// types of containers we'll want to test, covering interesting iterator types
namespace {
struct VectorContainer {
  template <typename... Args>
  using type = std::vector<Args...>;

  template <typename... Args>
  [[nodiscard]] static std::vector<Args...> sortedFrom(std::vector<Args...>&& in) {
    sortValues(in, Order::Ascending);
    return std::move(in);
  }

  static constexpr const char* Name = "Vector";
};

struct SetContainer {
  template <typename... Args>
  using type = std::set<Args...>;

  template <typename... Args>
  [[nodiscard]] static auto sortedFrom(std::vector<Args...>&& in) -> type<typename std::vector<Args...>::value_type> {
    type<typename std::vector<Args...>::value_type> out;
    std::move(in.begin(), in.end(), std::inserter(out, out.begin()));
    return out;
  }

  static constexpr const char* Name = "Set";
};
using AllContainerTypes = std::tuple<VectorContainer, SetContainer>;


template <class ValueType, class ContainerType>
struct LowerBound {
  size_t Quantity;

  mutable std::mt19937_64 rng{std::random_device{}()};

  void run(benchmark::State& state) const {
    runOpOnCopies<ValueType>(state, Quantity, Order::Random, BatchSize::CountBatch, [&](auto& Copy) {
      state.PauseTiming();
      auto needle = Copy[rng() % Copy.size()];
      auto in = ContainerType::sortedFrom(std::move(Copy));
      state.ResumeTiming();
      auto result = std::lower_bound(in.begin(), in.end(), needle);
      benchmark::DoNotOptimize(result);
    });
  }

  std::string name() const { return "BM_LowerBound" + ValueType::name() + "_" + ContainerType::Name + "_" + std::to_string(Quantity); }
};
} // namespace

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  makeCartesianProductBenchmark<LowerBound, AllValueTypes, AllContainerTypes>(Quantities);
  benchmark::RunSpecifiedBenchmarks();
}
