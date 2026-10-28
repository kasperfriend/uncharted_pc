// Stand-in for the GPU build's fmt (a submodule it links): the register-file headers include it
// for their debug strings, and tools/ci/linux-tests.sh compiles the tests that need those headers
// in a plain checkout, where the submodules are absent. Only the declarations those headers
// instantiate are here; nothing in the host tests formats anything through this.
#pragma once
#include <string>
#include <string_view>

namespace fmt {

struct format_parse_context {
    const char* position = nullptr;
    constexpr const char* begin() const {
        return position;
    }
    constexpr const char* end() const {
        return position;
    }
};

struct format_context {
    struct iterator {};
    constexpr iterator out() const {
        return {};
    }
};

inline std::string format(std::string_view, auto&&...) {
    return {};
}

template <typename OutputIt>
OutputIt format_to(OutputIt out, std::string_view, auto&&...) {
    return out;
}

template <typename T, typename Char = char, typename Enable = void>
struct formatter {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }
    auto format(const T&, format_context& ctx) const {
        return fmt::format_to(ctx.out(), "{}");
    }
};

template <typename... Args>
std::string runtime(std::string_view s, Args&&...) {
    return std::string{s};
}

} // namespace fmt
