#define FMT_HEADER_ONLY

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fmt/format.h>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <xtensor/xarray.hpp>

#include "src/simsoptpp/cache.h"

using Array = xt::xarray<double>;
using Clock = std::chrono::steady_clock;

namespace {

volatile double benchmark_sink = 0.0;

struct BenchmarkConfig {
    int ncoils = 16;
    int npoints = 400;
    int derivatives = 2;
    int warmup = 128;
    int iterations = 5000;
    int samples = 9;
};

struct SampleSummary {
    std::vector<double> samples_us;
    double median_us;
    double mean_us;
};

void touch(Array& array, double delta) {
    auto it = array.begin();
    *it += delta;
    benchmark_sink += *it;
}

SampleSummary summarize(std::vector<double> samples_us) {
    std::vector<double> sorted = samples_us;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t middle = sorted.size() / 2;
    const double median_us = sorted.size() % 2 == 0
        ? 0.5 * (sorted[middle - 1] + sorted[middle])
        : sorted[middle];
    const double mean_us =
        std::accumulate(samples_us.begin(), samples_us.end(), 0.0) / samples_us.size();
    return SampleSummary{std::move(samples_us), median_us, mean_us};
}

template<class MakeBody>
SampleSummary measure_samples(const BenchmarkConfig& config, MakeBody make_body) {
    std::vector<double> samples_us;
    samples_us.reserve(config.samples);

    for (int sample = 0; sample < config.samples; ++sample) {
        auto body = make_body();
        for (int i = 0; i < config.warmup; ++i) {
            body();
        }
        const auto start = Clock::now();
        for (int i = 0; i < config.iterations; ++i) {
            body();
        }
        const auto elapsed_us =
            std::chrono::duration<double, std::micro>(Clock::now() - start).count();
        samples_us.push_back(elapsed_us / config.iterations);
    }

    return summarize(std::move(samples_us));
}

void validate_config(const BenchmarkConfig& config) {
    if (config.ncoils <= 0 || config.npoints <= 0 || config.iterations <= 0 ||
        config.samples <= 0) {
        throw std::runtime_error(
            "ncoils, npoints, iterations, and samples must all be positive.");
    }
    if (config.warmup < 0) {
        throw std::runtime_error("warmup must be nonnegative.");
    }
    if (config.derivatives < 0 || config.derivatives > 2) {
        throw std::runtime_error("derivatives must be one of 0, 1, or 2.");
    }
}

BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(fmt::format("Missing value for {}", name));
            }
            ++i;
            return argv[i];
        };
        if (arg == "--ncoils") {
            config.ncoils = std::atoi(require_value("--ncoils"));
        } else if (arg == "--npoints") {
            config.npoints = std::atoi(require_value("--npoints"));
        } else if (arg == "--derivatives") {
            config.derivatives = std::atoi(require_value("--derivatives"));
        } else if (arg == "--warmup") {
            config.warmup = std::atoi(require_value("--warmup"));
        } else if (arg == "--iterations") {
            config.iterations = std::atoi(require_value("--iterations"));
        } else if (arg == "--samples") {
            config.samples = std::atoi(require_value("--samples"));
        } else {
            throw std::runtime_error(fmt::format("Unknown argument {}", arg));
        }
    }
    validate_config(config);
    return config;
}

void legacy_compute_body(
    Cache<Array>& cache,
    int ncoils,
    const std::vector<int>& dims_B,
    const std::vector<int>& dims_dB,
    const std::vector<int>& dims_ddB,
    int derivatives
) {
    for (int i = 0; i < ncoils; ++i) {
        touch(cache.get_or_create(fmt::format("B_{}", i), dims_B), 1.0);
        if (derivatives > 0) {
            touch(cache.get_or_create(fmt::format("dB_{}", i), dims_dB), 1.0);
        }
        if (derivatives > 1) {
            touch(cache.get_or_create(fmt::format("ddB_{}", i), dims_ddB), 1.0);
        }
    }

    for (int i = 0; i < ncoils; ++i) {
        touch(cache.get_or_create(fmt::format("B_{}", i), dims_B), 1.0);
        if (derivatives > 0) {
            touch(cache.get_or_create(fmt::format("dB_{}", i), dims_dB), 1.0);
        }
        if (derivatives > 1) {
            touch(cache.get_or_create(fmt::format("ddB_{}", i), dims_ddB), 1.0);
        }
    }

    for (int i = 0; i < ncoils; ++i) {
        touch(cache.get_or_create(fmt::format("B_{}", i), dims_B), 1.0);
    }
    if (derivatives > 0) {
        for (int i = 0; i < ncoils; ++i) {
            touch(cache.get_or_create(fmt::format("dB_{}", i), dims_dB), 1.0);
        }
    }
    if (derivatives > 1) {
        for (int i = 0; i < ncoils; ++i) {
            touch(cache.get_or_create(fmt::format("ddB_{}", i), dims_ddB), 1.0);
        }
    }
}

void indexed_compute_body(
    IndexedFieldCache<Array>& cache,
    int ncoils,
    int npoints,
    int derivatives
) {
    cache.prepare_magnetic_field_family(ncoils, npoints, derivatives);

    for (int i = 0; i < ncoils; ++i) {
        touch(cache.get(IndexedFieldCacheKind::B, i), 1.0);
        if (derivatives > 0) {
            touch(cache.get(IndexedFieldCacheKind::dB, i), 1.0);
        }
        if (derivatives > 1) {
            touch(cache.get(IndexedFieldCacheKind::ddB, i), 1.0);
        }
    }

    for (int i = 0; i < ncoils; ++i) {
        touch(cache.get(IndexedFieldCacheKind::B, i), 1.0);
        if (derivatives > 0) {
            touch(cache.get(IndexedFieldCacheKind::dB, i), 1.0);
        }
        if (derivatives > 1) {
            touch(cache.get(IndexedFieldCacheKind::ddB, i), 1.0);
        }
    }

    for (int i = 0; i < ncoils; ++i) {
        touch(cache.get(IndexedFieldCacheKind::B, i), 1.0);
    }
    if (derivatives > 0) {
        for (int i = 0; i < ncoils; ++i) {
            touch(cache.get(IndexedFieldCacheKind::dB, i), 1.0);
        }
    }
    if (derivatives > 1) {
        for (int i = 0; i < ncoils; ++i) {
            touch(cache.get(IndexedFieldCacheKind::ddB, i), 1.0);
        }
    }
}

void compat_canonical_body(
    IndexedFieldCache<Array>& indexed_cache,
    Cache<Array>& legacy_cache,
    int ncoils,
    const std::vector<int>& dims_B
) {
    for (int i = 0; i < ncoils; ++i) {
        touch(
            fieldcache_get_or_create_compat(
                indexed_cache,
                legacy_cache,
                std::string("B_") + std::to_string(i),
                dims_B
            ),
            1.0
        );
    }
}

void compat_unknown_body(
    IndexedFieldCache<Array>& indexed_cache,
    Cache<Array>& legacy_cache,
    int ncoils,
    const std::vector<int>& dims_B
) {
    for (int i = 0; i < ncoils; ++i) {
        touch(
            fieldcache_get_or_create_compat(
                indexed_cache,
                legacy_cache,
                std::string("legacy_") + std::to_string(i),
                dims_B
            ),
            1.0
        );
    }
}

SampleSummary measure_legacy_compute_bookkeeping(const BenchmarkConfig& config) {
    const int ncoils = config.ncoils;
    const int npoints = config.npoints;
    const int derivatives = config.derivatives;

    return measure_samples(config, [ncoils, npoints, derivatives] {
        Cache<Array> cache;
        std::vector<int> dims_B{npoints, 3};
        std::vector<int> dims_dB{npoints, 3, 3};
        std::vector<int> dims_ddB{npoints, 3, 3, 3};
        return [cache = std::move(cache),
                dims_B = std::move(dims_B),
                dims_dB = std::move(dims_dB),
                dims_ddB = std::move(dims_ddB),
                ncoils,
                derivatives]() mutable {
            legacy_compute_body(
                cache,
                ncoils,
                dims_B,
                dims_dB,
                dims_ddB,
                derivatives
            );
        };
    });
}

SampleSummary measure_indexed_compute_bookkeeping(const BenchmarkConfig& config) {
    const int ncoils = config.ncoils;
    const int npoints = config.npoints;
    const int derivatives = config.derivatives;

    return measure_samples(config, [ncoils, npoints, derivatives] {
        IndexedFieldCache<Array> cache;
        return [cache = std::move(cache), ncoils, npoints, derivatives]() mutable {
            indexed_compute_body(
                cache,
                ncoils,
                npoints,
                derivatives
            );
        };
    });
}

template<class CompatBody>
SampleSummary measure_compat_get_or_create(
    const BenchmarkConfig& config, CompatBody compat_body
) {
    const int ncoils = config.ncoils;
    const int npoints = config.npoints;

    return measure_samples(config, [ncoils, npoints, compat_body] {
        IndexedFieldCache<Array> indexed_cache;
        Cache<Array> legacy_cache;
        std::vector<int> dims_B{npoints, 3};
        return [indexed_cache = std::move(indexed_cache),
                legacy_cache = std::move(legacy_cache),
                dims_B = std::move(dims_B),
                ncoils,
                compat_body]() mutable {
            compat_body(indexed_cache, legacy_cache, ncoils, dims_B);
        };
    });
}

SampleSummary measure_compat_canonical_get_or_create(const BenchmarkConfig& config) {
    return measure_compat_get_or_create(config, compat_canonical_body);
}

SampleSummary measure_compat_unknown_get_or_create(const BenchmarkConfig& config) {
    return measure_compat_get_or_create(config, compat_unknown_body);
}

void print_samples(const std::vector<double>& values) {
    std::cout << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << values[i];
    }
    std::cout << "]";
}

void print_summary(const char* name, const SampleSummary& summary) {
    std::cout << "  \"" << name << "\": {\n";
    std::cout << "    \"median_us\": " << summary.median_us << ",\n";
    std::cout << "    \"mean_us\": " << summary.mean_us << ",\n";
    std::cout << "    \"samples_us\": ";
    print_samples(summary.samples_us);
    std::cout << "\n  }";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const BenchmarkConfig config = parse_args(argc, argv);
        const SampleSummary legacy_compute = measure_legacy_compute_bookkeeping(config);
        const SampleSummary indexed_compute = measure_indexed_compute_bookkeeping(config);
        const SampleSummary compat_canonical =
            measure_compat_canonical_get_or_create(config);
        const SampleSummary compat_unknown =
            measure_compat_unknown_get_or_create(config);

        std::cout << "{\n";
        std::cout << "  \"config\": {\n";
        std::cout << "    \"ncoils\": " << config.ncoils << ",\n";
        std::cout << "    \"npoints\": " << config.npoints << ",\n";
        std::cout << "    \"derivatives\": " << config.derivatives << ",\n";
        std::cout << "    \"warmup\": " << config.warmup << ",\n";
        std::cout << "    \"iterations\": " << config.iterations << ",\n";
        std::cout << "    \"samples\": " << config.samples << "\n";
        std::cout << "  },\n";
        print_summary("legacy_compute_bookkeeping", legacy_compute);
        std::cout << ",\n";
        print_summary("indexed_compute_bookkeeping", indexed_compute);
        std::cout << ",\n";
        print_summary("compat_canonical_get_or_create", compat_canonical);
        std::cout << ",\n";
        print_summary("compat_unknown_get_or_create", compat_unknown);
        std::cout << "\n}\n";

        if (benchmark_sink == 0.123456789) {
            std::cerr << benchmark_sink << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 2;
    }
}
