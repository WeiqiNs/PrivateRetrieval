#ifndef PRIVRET_SEARCH_HPP
#define PRIVRET_SEARCH_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "format.hpp"

namespace privret{
    struct Hit{
        std::string doc;
        std::int64_t score;
        friend bool operator==(const Hit&, const Hit&) = default;
    };

    struct Ranking{
        std::string query;
        std::vector<Hit> hits;
        friend bool operator==(const Ranking&, const Ranking&) = default;
    };

    using Run = std::vector<Ranking>;

    template <class D, class Q>
    struct SearchInput{
        const Table<D>& docs;
        const Table<Q>& queries;
        std::size_t top_k;
    };

    class DecryptError : public std::runtime_error{
    public:
        using std::runtime_error::runtime_error;
    };

    inline constexpr std::string_view run_tag = "privret";

    [[nodiscard]] inline bool ranks_before(const Hit& a, const Hit& b){
        return a.score != b.score ? a.score > b.score : a.doc < b.doc;
    }

    [[nodiscard]] inline std::vector<Hit> top_k(std::vector<Hit> hits, const std::size_t k){
        const auto kept = std::min(k, hits.size());
        std::partial_sort(hits.begin(), hits.begin() + static_cast<std::ptrdiff_t>(kept), hits.end(), ranks_before);
        hits.resize(kept);
        return hits;
    }

    template <class D, class Q, class Score>
    [[nodiscard]] Run rank(const SearchInput<D, Q>& input, Score score){
        const auto& [docs, queries, k] = input;
        if (docs.dim != queries.dim)
            throw FormatError(std::format("documents have dim {} but queries have dim {}", docs.dim, queries.dim));
        Run run;
        run.reserve(queries.ids.size());
        for (std::size_t q = 0; q < queries.ids.size(); ++q){
            std::vector<Hit> hits;
            hits.reserve(docs.ids.size());
            for (std::size_t d = 0; d < docs.ids.size(); ++d) hits.push_back({docs.ids[d], score(q, d)});
            run.push_back({queries.ids[q], top_k(std::move(hits), k)});
        }
        return run;
    }

    [[nodiscard]] inline std::int64_t dot(const Embedding& a, const Embedding& b){
        std::int64_t total = 0;
        for (std::size_t i = 0; i < a.size(); ++i) total += std::int64_t{a[i]} * b[i];
        return total;
    }

    [[nodiscard]] inline std::uint64_t isqrt(const std::uint64_t x){
        auto r = static_cast<std::uint64_t>(std::sqrt(static_cast<long double>(x)));
        while (r * r > x) --r;
        while ((r + 1) * (r + 1) <= x) ++r;
        return r;
    }

    [[nodiscard]] inline Run search_plain(const SearchInput<Embedding, Embedding>& input){
        return rank(input, [&](const std::size_t q, const std::size_t d){
            return dot(input.queries.records[q], input.docs.records[d]);
        });
    }

    [[nodiscard]] inline rbp::DlogTable<Curve> score_table(const SearchInput<Ct, Sk>& input){
        const auto bound = static_cast<std::int64_t>(isqrt(input.docs.max_norm_sq * input.queries.max_norm_sq));
        return {IPFE::OPT::base<Curve>(), -bound, bound};
    }

    [[nodiscard]] inline Run search_encrypted(const SearchInput<Ct, Sk>& input, const rbp::DlogTable<Curve>& table){
        return rank(input, [&](const std::size_t q, const std::size_t d){
            const auto score = IPFE::OPT::dec(table, input.queries.records[q], input.docs.records[d]);
            if (!score)
                throw DecryptError(std::format(
                    "query {} and doc {} decrypt outside the score range; were they made from the same master key?",
                    input.queries.ids[q], input.docs.ids[d]
                ));
            return *score;
        });
    }

    [[nodiscard]] inline Run merge(const std::vector<Run>& runs, const std::size_t k){
        Run merged;
        std::unordered_map<std::string_view, std::size_t> position;
        for (const auto& run : runs){
            for (const auto& [query, hits] : run){
                const auto [slot, added] = position.try_emplace(query, merged.size());
                if (added) merged.push_back({query, {}});
                auto& target = merged[slot->second].hits;
                target.insert(target.end(), hits.begin(), hits.end());
            }
        }
        for (auto& [query, hits] : merged){
            std::unordered_set<std::string_view> seen;
            for (const auto& hit : hits)
                if (!seen.insert(hit.doc).second) throw FormatError(std::format("query {} lists doc {} twice", query, hit.doc));
            hits = top_k(std::move(hits), k);
        }
        return merged;
    }

    [[nodiscard]] inline std::string format_run(const Run& run){
        std::string text;
        for (const auto& [query, hits] : run)
            for (std::size_t i = 0; i < hits.size(); ++i)
                text += std::format("{} Q0 {} {} {} {}\n", query, hits[i].doc, i + 1, hits[i].score, run_tag);
        return text;
    }

    namespace detail{
        [[nodiscard]] inline std::vector<std::string_view> split(const std::string_view text, const char separator){
            std::vector<std::string_view> parts;
            std::size_t start = 0;
            for (auto end = text.find(separator); end != std::string_view::npos; end = text.find(separator, start)){
                parts.push_back(text.substr(start, end - start));
                start = end + 1;
            }
            parts.push_back(text.substr(start));
            return parts;
        }
    }

    [[nodiscard]] inline Run parse_run(const std::string_view text){
        Run run;
        std::unordered_map<std::string_view, std::size_t> position;
        auto lines = detail::split(text, '\n');
        if (lines.back().empty()) lines.pop_back();
        for (std::size_t n = 0; n < lines.size(); ++n){
            const auto fields = detail::split(lines[n], ' ');
            const auto well_formed = fields.size() == 6 && !fields[0].empty() && fields[1] == "Q0" && !fields[2].empty()
                && parse_integer<std::int64_t>(fields[3]);
            const auto score = well_formed ? parse_integer<std::int64_t>(fields[4]) : std::nullopt;
            if (!score)
                throw FormatError(std::format("run line {} is not 'query Q0 doc rank score tag' with integer rank and score", n + 1));
            const auto [slot, added] = position.try_emplace(fields[0], run.size());
            if (added) run.push_back({std::string(fields[0]), {}});
            run[slot->second].hits.push_back({std::string(fields[2]), *score});
        }
        return run;
    }
}

#endif
