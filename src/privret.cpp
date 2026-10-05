#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <format>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <privret/format.hpp>
#include <privret/search.hpp>

namespace{
    using namespace privret;

    class UsageError : public std::runtime_error{
    public:
        using std::runtime_error::runtime_error;
    };

    struct Args{
        std::map<std::string, std::string, std::less<>> options;
        std::vector<std::string> inputs;

        [[nodiscard]] const std::string& require(const std::string_view flag) const{
            const auto found = options.find(flag);
            if (found == options.end()) throw UsageError(std::format("missing {}", flag));
            return found->second;
        }

        [[nodiscard]] std::optional<std::string> find(const std::string_view flag) const{
            const auto found = options.find(flag);
            return found == options.end() ? std::nullopt : std::optional(found->second);
        }
    };

    struct Sealing{
        std::string_view command;
        std::string_view items;
        std::string_view ms_per_item;
    };

    constexpr std::string_view usage = R"(usage:
  privret setup   --dim D --msk PATH
  privret encrypt --msk PATH --vectors PATH --out PATH
  privret keygen  --msk PATH --vectors PATH --out PATH [--max-queries Q]
  privret search  --corpus PATH --keys PATH --top-k K --out PATH [--shard i/N]
  privret merge   --top-k K --out PATH RUN...
  privret plain   --docs PATH --queries PATH --top-k K --out PATH [--max-queries Q]
)";

    [[nodiscard]] Args parse_args(const int argc, char** argv){
        Args args;
        for (int i = 2; i < argc; ++i){
            const std::string_view token = argv[i];
            if (!token.starts_with("--")){
                args.inputs.emplace_back(token);
                continue;
            }
            if (i + 1 == argc) throw UsageError(std::format("{} needs a value", token));
            if (!args.options.emplace(token, argv[++i]).second) throw UsageError(std::format("{} given twice", token));
        }
        return args;
    }

    [[nodiscard]] std::size_t parse_count(const std::string_view text){
        const auto value = parse_integer<std::size_t>(text);
        if (!value || *value == 0) throw UsageError(std::format("'{}' is not a positive integer", text));
        return *value;
    }

    void report(const std::string_view command, const std::initializer_list<std::pair<std::string_view, double>> fields){
        auto line = std::format(R"({{"command":"{}")", command);
        for (const auto& [name, value] : fields) line += std::format(R"(,"{}":{})", name, value);
        std::cerr << line + "}\n";
    }

    class Stopwatch{
    public:
        [[nodiscard]] double seconds() const{
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
        }

    private:
        std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    };

    [[nodiscard]] double ms_per(const double seconds, const std::size_t count){
        return count == 0 ? 0.0 : 1000.0 * seconds / static_cast<double>(count);
    }

    [[nodiscard]] Table<Embedding> query_vectors(const Args& args, const std::string_view flag){
        auto queries = decode_table<Embedding>(read_file(args.require(flag)));
        if (const auto limit = args.find("--max-queries")) return first(queries, parse_count(*limit));
        return queries;
    }

    template <class Make>
    int seal(const Args& args, const Table<Embedding>& vectors, const Sealing& sealing, Make make){
        const auto msk = decode_msk(read_file(args.require("--msk")));
        if (vectors.dim != msk.a.cols())
            throw FormatError(std::format("vectors have dim {} but the master key has dim {}", vectors.dim, msk.a.cols()));
        const Stopwatch clock;
        const auto sealed = map_records(vectors, [&](const Embedding& v){ return make(msk, widen(v)); });
        const auto seconds = clock.seconds();
        const auto bytes = encode_table(sealed);
        write_file(args.require("--out"), bytes);
        const auto count = vectors.ids.size();
        report(sealing.command, {
            {"dim", vectors.dim}, {sealing.items, count}, {"seconds", seconds}, {sealing.ms_per_item, ms_per(seconds, count)},
            {"bytes", bytes.size()}
        });
        return 0;
    }

    int setup(const Args& args){
        const auto dim = parse_count(args.require("--dim"));
        if (dim > max_dim) throw UsageError(std::format("--dim must be at most {}", max_dim));
        const std::filesystem::path path = args.require("--msk");
        const Stopwatch clock;
        const auto msk = IPFE::OPT::setup<Curve>(dim);
        const auto seconds = clock.seconds();
        write_file(path, std::string_view{});
        std::filesystem::permissions(
            path, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write, std::filesystem::perm_options::replace
        );
        write_file(path, encode_msk(msk));
        report("setup", {{"dim", dim}, {"seconds", seconds}});
        return 0;
    }

    int encrypt(const Args& args){
        return seal(
            args, decode_table<Embedding>(read_file(args.require("--vectors"))), {"encrypt", "docs", "ms_per_doc"},
            [](const Msk& msk, const IPFE::IntVec& v){ return IPFE::OPT::enc(msk, v); }
        );
    }

    int keygen(const Args& args){
        return seal(
            args, query_vectors(args, "--vectors"), {"keygen", "queries", "ms_per_query"},
            [](const Msk& msk, const IPFE::IntVec& v){ return IPFE::OPT::keygen(msk, v); }
        );
    }

    int search(const Args& args){
        const auto top_k = parse_count(args.require("--top-k"));
        const auto shard_text = args.find("--shard");
        const auto shard = shard_text ? parse_shard(*shard_text) : Shard{};
        const Stopwatch load_clock;
        const auto corpus = decode_table<Ct>(read_file(args.require("--corpus")), shard);
        const auto keys = decode_table<Sk>(read_file(args.require("--keys")));
        const auto load_seconds = load_clock.seconds();
        const SearchInput<Ct, Sk> input{.docs = corpus, .queries = keys, .top_k = top_k};
        const Stopwatch table_clock;
        const auto table = score_table(input);
        const auto table_seconds = table_clock.seconds();
        const Stopwatch clock;
        const auto run = search_encrypted(input, table);
        const auto seconds = clock.seconds();
        write_file(args.require("--out"), format_run(run));
        const auto pairs = corpus.ids.size() * keys.ids.size();
        report("search", {
            {"dim", corpus.dim}, {"shard_index", shard.index}, {"shard_count", shard.count}, {"docs", corpus.ids.size()},
            {"queries", keys.ids.size()}, {"pairs", pairs}, {"load_seconds", load_seconds}, {"table_ms", 1000.0 * table_seconds},
            {"seconds", seconds}, {"ms_per_pair", ms_per(seconds, pairs)}
        });
        return 0;
    }

    int merge_runs(const Args& args){
        const auto top_k = parse_count(args.require("--top-k"));
        if (args.inputs.empty()) throw UsageError("merge needs at least one run file");
        std::vector<privret::Run> runs;
        for (const auto& path : args.inputs){
            const auto bytes = read_file(path);
            runs.push_back(parse_run({reinterpret_cast<const char*>(bytes.data()), bytes.size()}));
        }
        write_file(args.require("--out"), format_run(merge(runs, top_k)));
        return 0;
    }

    int plain(const Args& args){
        const auto top_k = parse_count(args.require("--top-k"));
        const auto docs = decode_table<Embedding>(read_file(args.require("--docs")));
        const auto queries = query_vectors(args, "--queries");
        const Stopwatch clock;
        const auto run = search_plain({.docs = docs, .queries = queries, .top_k = top_k});
        const auto seconds = clock.seconds();
        write_file(args.require("--out"), format_run(run));
        report("plain", {{"docs", docs.ids.size()}, {"queries", queries.ids.size()}, {"seconds", seconds}});
        return 0;
    }

    const std::map<std::string_view, int (*)(const Args&)> commands{
        {"setup", setup}, {"encrypt", encrypt}, {"keygen", keygen}, {"search", search}, {"merge", merge_runs}, {"plain", plain},
    };
}

int main(const int argc, char** argv){
    try{
        if (argc < 2) throw UsageError("missing command");
        const auto command = commands.find(argv[1]);
        if (command == commands.end()) throw UsageError(std::format("unknown command '{}'", argv[1]));
        return command->second(parse_args(argc, argv));
    }
    catch (const UsageError& error){
        std::cerr << "privret: " << error.what() << '\n' << usage;
        return 2;
    }
    catch (const std::exception& error){
        std::cerr << "privret: " << error.what() << '\n';
        return 1;
    }
}
