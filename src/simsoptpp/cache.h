#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <xtensor/xarray.hpp>
//#include <fmt/core.h>
//#include <fmt/format.h>
//#include <fmt/ranges.h>
#include "cachedarray.h"


using std::string;
using std::vector;

enum class IndexedFieldCacheKind {
    B,
    dB,
    ddB,
    A,
    dA,
    ddA
};

struct IndexedFieldCacheKey {
    IndexedFieldCacheKind kind;
    std::size_t index;
};

inline std::optional<IndexedFieldCacheKey> parse_indexed_field_cache_key(const string& key) {
    static constexpr std::pair<std::string_view, IndexedFieldCacheKind> prefixes[] = {
        {"ddB", IndexedFieldCacheKind::ddB},
        {"ddA", IndexedFieldCacheKind::ddA},
        {"dB", IndexedFieldCacheKind::dB},
        {"dA", IndexedFieldCacheKind::dA},
        {"B", IndexedFieldCacheKind::B},
        {"A", IndexedFieldCacheKind::A},
    };

    for (const auto& [prefix, kind] : prefixes) {
        if (key.size() <= prefix.size() + 1) {
            continue;
        }
        if (key.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        if (key[prefix.size()] != '_') {
            continue;
        }

        std::size_t index = 0;
        for (std::size_t i = prefix.size() + 1; i < key.size(); ++i) {
            const char digit = key[i];
            if (digit < '0' || digit > '9') {
                return std::nullopt;
            }
            index = index * 10 + static_cast<std::size_t>(digit - '0');
        }
        return IndexedFieldCacheKey{kind, index};
    }

    return std::nullopt;
}

inline bool indexed_field_cache_dims_match(
    IndexedFieldCacheKind kind, const vector<int>& dims
) {
    switch (kind) {
        case IndexedFieldCacheKind::B:
        case IndexedFieldCacheKind::A:
            return dims.size() == 2 && dims[1] == 3;
        case IndexedFieldCacheKind::dB:
        case IndexedFieldCacheKind::dA:
            return dims.size() == 3 && dims[1] == 3 && dims[2] == 3;
        case IndexedFieldCacheKind::ddB:
        case IndexedFieldCacheKind::ddA:
            return dims.size() == 4 && dims[1] == 3 && dims[2] == 3 && dims[3] == 3;
    }
    throw std::logic_error("Unknown indexed field cache kind.");
}

template<class Array>
class Cache {
    private:
        std::map<string, CachedArray<Array>> cache;
    public:
        bool get_status(string key) const {
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found
                return false;
            }
            if(!(loc->second.status)){ // needs recomputing
                return false;
            }
            return true;
        }
        Array& get_or_create(string key, vector<int> dims){
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>(dims)))).first; 
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            } else if(loc->second.data.shape(0) != dims[0]) { // key found but not the right number of points
                loc->second = CachedArray<Array>(xt::zeros<double>(dims));
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            } else {
                //fmt::print("Existing array found for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            }
            loc->second.status = true;
            return loc->second.data;
        }

        Array& get_or_create_and_fill(string key, vector<int> dims, std::function<void(Array&)> impl) {
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>(dims)))).first; 
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            } else if(loc->second.data.shape(0) != dims[0]) { // key found but not the right number of points
                loc->second = CachedArray<Array>(xt::zeros<double>(dims));
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            }
            if(!(loc->second.status)){ // needs recomputing
                //fmt::print("Fill array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
                impl(loc->second.data);
                loc->second.status = true;
            }
            return loc->second.data;
        }

        void invalidate_cache(){
            for (auto it = cache.begin(); it != cache.end(); ++it) {
                it->second.status = false;
            }
        }
};

template<class Array>
class IndexedFieldCache {
    private:
        using SlotVector = std::vector<CachedArray<Array>>;

        SlotVector B_cache;
        SlotVector dB_cache;
        SlotVector ddB_cache;
        SlotVector A_cache;
        SlotVector dA_cache;
        SlotVector ddA_cache;

        SlotVector& slots(IndexedFieldCacheKind kind) {
            switch (kind) {
                case IndexedFieldCacheKind::B:
                    return B_cache;
                case IndexedFieldCacheKind::dB:
                    return dB_cache;
                case IndexedFieldCacheKind::ddB:
                    return ddB_cache;
                case IndexedFieldCacheKind::A:
                    return A_cache;
                case IndexedFieldCacheKind::dA:
                    return dA_cache;
                case IndexedFieldCacheKind::ddA:
                    return ddA_cache;
            }
            throw std::logic_error("Unknown indexed field cache kind.");
        }

        const SlotVector& slots(IndexedFieldCacheKind kind) const {
            switch (kind) {
                case IndexedFieldCacheKind::B:
                    return B_cache;
                case IndexedFieldCacheKind::dB:
                    return dB_cache;
                case IndexedFieldCacheKind::ddB:
                    return ddB_cache;
                case IndexedFieldCacheKind::A:
                    return A_cache;
                case IndexedFieldCacheKind::dA:
                    return dA_cache;
                case IndexedFieldCacheKind::ddA:
                    return ddA_cache;
            }
            throw std::logic_error("Unknown indexed field cache kind.");
        }

        static Array make_array(IndexedFieldCacheKind kind, int npoints) {
            switch (kind) {
                case IndexedFieldCacheKind::B:
                case IndexedFieldCacheKind::A:
                    return xt::zeros<double>({npoints, 3});
                case IndexedFieldCacheKind::dB:
                case IndexedFieldCacheKind::dA:
                    return xt::zeros<double>({npoints, 3, 3});
                case IndexedFieldCacheKind::ddB:
                case IndexedFieldCacheKind::ddA:
                    return xt::zeros<double>({npoints, 3, 3, 3});
            }
            throw std::logic_error("Unknown indexed field cache kind.");
        }

        CachedArray<Array>& ensure_slot(IndexedFieldCacheKind kind, std::size_t index, int npoints) {
            auto& cache_for_kind = slots(kind);
            while (cache_for_kind.size() <= index) {
                cache_for_kind.emplace_back(make_array(kind, npoints));
            }
            auto& slot = cache_for_kind[index];
            if (slot.data.shape(0) != npoints) {
                slot = CachedArray<Array>(make_array(kind, npoints));
            }
            slot.status = true;
            return slot;
        }

        void prepare_kind(IndexedFieldCacheKind kind, std::size_t count, int npoints) {
            for (std::size_t i = 0; i < count; ++i) {
                ensure_slot(kind, i, npoints);
            }
        }

    public:
        Array& get_or_create(IndexedFieldCacheKind kind, std::size_t index, int npoints) {
            return ensure_slot(kind, index, npoints).data;
        }

        Array& get(IndexedFieldCacheKind kind, std::size_t index) {
            return slots(kind)[index].data;
        }

        const Array& get(IndexedFieldCacheKind kind, std::size_t index) const {
            return slots(kind)[index].data;
        }

        bool get_status(IndexedFieldCacheKind kind, std::size_t index) const {
            const auto& cache_for_kind = slots(kind);
            if (index >= cache_for_kind.size()) {
                return false;
            }
            return cache_for_kind[index].status;
        }

        bool has_slot(IndexedFieldCacheKind kind, std::size_t index) const {
            return index < slots(kind).size();
        }

        void prepare_magnetic_field_family(std::size_t count, int npoints, int derivatives) {
            prepare_kind(IndexedFieldCacheKind::B, count, npoints);
            if (derivatives > 0) {
                prepare_kind(IndexedFieldCacheKind::dB, count, npoints);
            }
            if (derivatives > 1) {
                prepare_kind(IndexedFieldCacheKind::ddB, count, npoints);
            }
        }

        void prepare_vector_potential_family(std::size_t count, int npoints, int derivatives) {
            prepare_kind(IndexedFieldCacheKind::A, count, npoints);
            if (derivatives > 0) {
                prepare_kind(IndexedFieldCacheKind::dA, count, npoints);
            }
            if (derivatives > 1) {
                prepare_kind(IndexedFieldCacheKind::ddA, count, npoints);
            }
        }

        void invalidate_cache() {
            for (auto kind : {
                IndexedFieldCacheKind::B,
                IndexedFieldCacheKind::dB,
                IndexedFieldCacheKind::ddB,
                IndexedFieldCacheKind::A,
                IndexedFieldCacheKind::dA,
                IndexedFieldCacheKind::ddA,
            }) {
                auto& cache_for_kind = slots(kind);
                for (auto& slot : cache_for_kind) {
                    slot.status = false;
                }
            }
        }
};

template<class Array>
inline Array& fieldcache_get_or_create_compat(
    IndexedFieldCache<Array>& indexed_cache,
    Cache<Array>& legacy_cache,
    const string& key,
    const vector<int>& dims
) {
    if (auto cache_key = parse_indexed_field_cache_key(key);
        cache_key.has_value() && indexed_field_cache_dims_match(cache_key->kind, dims)) {
        return indexed_cache.get_or_create(cache_key->kind, cache_key->index, dims[0]);
    }
    return legacy_cache.get_or_create(key, dims);
}

template<class Array>
inline bool fieldcache_get_status_compat(
    const IndexedFieldCache<Array>& indexed_cache,
    const Cache<Array>& legacy_cache,
    const string& key
) {
    if (auto cache_key = parse_indexed_field_cache_key(key); cache_key.has_value()) {
        if (indexed_cache.has_slot(cache_key->kind, cache_key->index)) {
            return indexed_cache.get_status(cache_key->kind, cache_key->index);
        }
    }
    return legacy_cache.get_status(key);
}
