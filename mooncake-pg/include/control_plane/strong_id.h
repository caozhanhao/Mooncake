#ifndef MOONCAKE_PG_CONTROL_PLANE_STRONG_ID_H
#define MOONCAKE_PG_CONTROL_PLANE_STRONG_ID_H

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

namespace mooncake {

// StrongId<Tag, Underlying> -- a zero-overhead "newtype" wrapper around an
// integer.  Two StrongIds with different tags are unrelated types even if they
// share the same underlying representation, which prevents accidentally using
// an InGroupRank where a GlobalRank is expected (and vice versa).
//
// The wrapper is intentionally simple: trivially copyable, standard-layout,
// and small enough that optimized builds compile to the same machine code as
// a raw integer.
template <typename Tag, typename Underlying = int32_t>
struct StrongId {
    // Public underlying value.  Kept public so that device code and plain
    // C-style boundaries can read it directly without an accessor function.
    // The tag (not access control) is what prevents mixing InGroupRank and
    // GlobalRank.
    Underlying value{};

    // No user-provided constructor: StrongId is an aggregate, which makes it
    // reflectable by struct_pack and prevents implicit conversion from a plain
    // integer (GlobalRank r = 5 is ill-formed).  Construct with braces:
    //   GlobalRank r{5};

    // Explicit conversion back to the underlying integer.  Kept for callers
    // that prefer the explicit cast style.
    explicit constexpr operator Underlying() const noexcept { return value; }

    // Convenience conversion to any arithmetic type.
    template <typename T>
    constexpr T to() const noexcept {
        return static_cast<T>(value);
    }

    // Convenience formatting so callers do not have to unwrap the value
    // manually for logs or error messages.
    std::string toString() const { return std::to_string(value); }

    // Comparisons against another StrongId with the same tag.
    constexpr bool operator==(StrongId other) const noexcept {
        return value == other.value;
    }
    constexpr bool operator!=(StrongId other) const noexcept {
        return value != other.value;
    }
    constexpr bool operator<(StrongId other) const noexcept {
        return value < other.value;
    }
    constexpr bool operator<=(StrongId other) const noexcept {
        return value <= other.value;
    }
    constexpr bool operator>(StrongId other) const noexcept {
        return value > other.value;
    }
    constexpr bool operator>=(StrongId other) const noexcept {
        return value >= other.value;
    }

    // Comparisons against plain integers (for bounds checks only).  These do
    // NOT allow implicit conversion from an integer to a StrongId, but they do
    // let you write lr < kMaxNumRanks or gr == 0 naturally.
    template <typename Integral,
              typename = std::enable_if_t<std::is_integral<Integral>::value>>
    constexpr bool operator==(Integral rhs) const noexcept {
        return value == static_cast<Underlying>(rhs);
    }
    template <typename Integral,
              typename = std::enable_if_t<std::is_integral<Integral>::value>>
    constexpr bool operator!=(Integral rhs) const noexcept {
        return value != static_cast<Underlying>(rhs);
    }
    template <typename Integral,
              typename = std::enable_if_t<std::is_integral<Integral>::value>>
    constexpr bool operator<(Integral rhs) const noexcept {
        return value < static_cast<Underlying>(rhs);
    }
    template <typename Integral,
              typename = std::enable_if_t<std::is_integral<Integral>::value>>
    constexpr bool operator<=(Integral rhs) const noexcept {
        return value <= static_cast<Underlying>(rhs);
    }
    template <typename Integral,
              typename = std::enable_if_t<std::is_integral<Integral>::value>>
    constexpr bool operator>(Integral rhs) const noexcept {
        return value > static_cast<Underlying>(rhs);
    }
    template <typename Integral,
              typename = std::enable_if_t<std::is_integral<Integral>::value>>
    constexpr bool operator>=(Integral rhs) const noexcept {
        return value >= static_cast<Underlying>(rhs);
    }

    // Increment operators so that strongly-typed for-loops work naturally:
    //   for (GlobalRank r{0}; r < GlobalRank{max_world_size}; ++r) { ... }
    constexpr StrongId& operator++() noexcept {
        ++value;
        return *this;
    }
    constexpr StrongId operator++(int) noexcept {
        StrongId tmp = *this;
        ++value;
        return tmp;
    }
};

// Stream output (logs, debugging).
template <typename Tag, typename Underlying>
inline std::ostream& operator<<(std::ostream& os,
                                StrongId<Tag, Underlying> id) {
    os << id.value;
    return os;
}

// Read the underlying value without exposing the member directly.
template <typename Tag, typename Underlying>
constexpr Underlying toUnderlying(StrongId<Tag, Underlying> id) noexcept {
    return id.value;
}

// Traits helper to extract tag/underlying from a StrongId type.
template <typename T>
struct StrongIdTraits;

template <typename Tag, typename Underlying>
struct StrongIdTraits<StrongId<Tag, Underlying>> {
    using TagType = Tag;
    using UnderlyingType = Underlying;
};

// IndexRange<IndexTag> -- a lightweight, strongly-typed integer range that
// lets range-based for loops keep their index namespace:
//
//   for (InGroupRank j : makeIndexRange<InGroupRank>(0, group_size)) { ... }
//
// In optimized builds this compiles to the same code as a raw int loop.
template <typename IndexTag, typename Underlying = int32_t>
class IndexRange {
   public:
    using IndexType = StrongId<IndexTag, Underlying>;

    class Iterator {
       public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = IndexType;
        using difference_type = std::ptrdiff_t;
        using pointer = const IndexType*;
        using reference = IndexType;

        explicit constexpr Iterator(IndexType cur) noexcept : cur_(cur) {}

        constexpr IndexType operator*() const noexcept { return cur_; }

        constexpr Iterator& operator++() noexcept {
            ++cur_;
            return *this;
        }
        constexpr Iterator operator++(int) noexcept {
            Iterator tmp = *this;
            ++cur_;
            return tmp;
        }

        constexpr bool operator==(Iterator other) const noexcept {
            return cur_ == other.cur_;
        }
        constexpr bool operator!=(Iterator other) const noexcept {
            return cur_ != other.cur_;
        }

       private:
        IndexType cur_;
    };

    explicit constexpr IndexRange(IndexType begin, IndexType end) noexcept
        : begin_(begin), end_(end) {}

    constexpr Iterator begin() const noexcept { return Iterator{begin_}; }
    constexpr Iterator end() const noexcept { return Iterator{end_}; }

   private:
    IndexType begin_;
    IndexType end_;
};

// IndexedVector<T, IndexTag> -- a std::vector wrapper whose index operator
// accepts only StrongId<IndexTag>.  Indexing with a raw integer or with a
// StrongId carrying a different tag is a compile-time error.
//
// It also satisfies struct_pack's container concept, so it can be used
// directly in RPC structs while serializing exactly like a std::vector.
//
// Use this for arrays whose index namespace matters, e.g.
//   IndexedVector<bool, GlobalRankTag>        link_status;  //
//   GlobalRank-indexed IndexedVector<GlobalRank, InGroupRankTag> rank_order; //
//   InGroupRank -> GlobalRank
template <typename T, typename IndexTag, typename Underlying = int32_t>
class IndexedVector {
    static_assert(std::is_same<Underlying, int32_t>::value,
                  "IndexedVector currently supports int32_t indices only");

   public:
    using IndexType = StrongId<IndexTag, Underlying>;
    using value_type = T;

    IndexedVector() = default;
    explicit IndexedVector(size_t size) : data_(size) {}
    IndexedVector(size_t size, const T& value) : data_(size, value) {}
    IndexedVector(std::initializer_list<T> init) : data_(init) {}

    // Only the declared index type may access elements.
    T& operator[](IndexType idx) {
        return data_[static_cast<size_t>(idx.value)];
    }
    const T& operator[](IndexType idx) const {
        return data_[static_cast<size_t>(idx.value)];
    }

    // Delete every other indexing overload so that mistakes fail at compile
    // time rather than runtime.
    template <typename Other>
    T& operator[](Other) = delete;
    template <typename Other>
    const T& operator[](Other) const = delete;

    size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }
    bool hasIndex(IndexType idx) const noexcept {
        return static_cast<size_t>(idx.value) < data_.size();
    }
    void resize(size_t n) { data_.resize(n); }
    void resize(size_t n, const T& value) { data_.resize(n, value); }
    void reserve(size_t n) { data_.reserve(n); }
    void assign(size_t n, const T& value) { data_.assign(n, value); }
    template <typename InputIt, typename = std::enable_if_t<!std::is_integral<
                                    typename std::decay<InputIt>::type>::value>>
    void assign(InputIt first, InputIt last) {
        data_.assign(first, last);
    }
    void push_back(const T& value) { data_.push_back(value); }
    template <typename... Args>
    decltype(auto) emplace_back(Args&&... args) {
        return data_.emplace_back(std::forward<Args>(args)...);
    }
    void clear() { data_.clear(); }
    void shrink_to_fit() { data_.shrink_to_fit(); }

    T& front() { return data_.front(); }
    const T& front() const { return data_.front(); }
    T& back() { return data_.back(); }
    const T& back() const { return data_.back(); }

    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    // Strongly-typed index range [0, size).  Use this to avoid manual
    // IndexType construction in range-based for loops:
    //   for (GlobalRank r : vec.indices()) { ... }
    IndexRange<IndexTag, Underlying> indices() const {
        return IndexRange<IndexTag, Underlying>(
            IndexType{static_cast<Underlying>(0)},
            IndexType{static_cast<Underlying>(data_.size())});
    }

    typename std::vector<T>::iterator begin() { return data_.begin(); }
    typename std::vector<T>::iterator end() { return data_.end(); }
    typename std::vector<T>::const_iterator begin() const {
        return data_.begin();
    }
    typename std::vector<T>::const_iterator end() const { return data_.end(); }

   private:
    std::vector<T> data_;
};

// Create a range [begin, end) of strongly-typed indices.
template <typename IndexType>
constexpr IndexRange<typename StrongIdTraits<IndexType>::TagType,
                     typename StrongIdTraits<IndexType>::UnderlyingType>
makeIndexRange(
    typename StrongIdTraits<IndexType>::UnderlyingType begin,
    typename StrongIdTraits<IndexType>::UnderlyingType end) noexcept {
    using Tag = typename StrongIdTraits<IndexType>::TagType;
    using Underlying = typename StrongIdTraits<IndexType>::UnderlyingType;
    return IndexRange<Tag, Underlying>(StrongId<Tag, Underlying>{begin},
                                       StrongId<Tag, Underlying>{end});
}

}  // namespace mooncake

// std::hash support so that StrongId can be used in unordered containers.
namespace std {

template <typename Tag, typename Underlying>
struct hash<mooncake::StrongId<Tag, Underlying>> {
    size_t operator()(
        const mooncake::StrongId<Tag, Underlying>& id) const noexcept {
        return std::hash<Underlying>{}(static_cast<Underlying>(id));
    }
};

}  // namespace std

#endif  // MOONCAKE_PG_CONTROL_PLANE_STRONG_ID_H
