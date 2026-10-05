#ifndef PRIVRET_FORMAT_HPP
#define PRIVRET_FORMAT_HPP

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
#include <PFE/ipfe_opt.hpp>

namespace privret{
    using Curve = rbp::BLS12_381;
    using Msk = IPFE::OPT::Msk<Curve>;
    using Ct = IPFE::OPT::Ct<Curve>;
    using Sk = IPFE::OPT::Sk<Curve>;
    using Embedding = std::vector<std::int8_t>;

    inline constexpr std::uint32_t format_version = 1;
    inline constexpr std::size_t max_dim = 4096;
    inline constexpr std::int64_t int8_limit = 127;

    class FormatError : public std::runtime_error{
    public:
        using std::runtime_error::runtime_error;
    };

    struct Shard{
        std::size_t index = 0;
        std::size_t count = 1;
        friend bool operator==(const Shard&, const Shard&) = default;
    };

    template <class Record>
    struct Table{
        std::vector<std::string> ids;
        std::size_t dim;
        std::uint64_t max_norm_sq;
        std::vector<Record> records;
    };

    template <class Record>
    struct Codec;

    namespace detail{
        inline constexpr std::string_view msk_magic = "PRMK";

        template <std::unsigned_integral U>
        void put(rbp::Bytes& out, const U value){
            for (std::size_t i = 0; i < sizeof(U); ++i) out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
        }

        inline void put_preamble(rbp::Bytes& out, const std::string_view magic, const std::size_t dim){
            out.insert(out.end(), magic.begin(), magic.end());
            put(out, format_version);
            put(out, static_cast<std::uint32_t>(dim));
        }

        inline void check_dim(const std::size_t dim){
            if (dim < 1 || dim > max_dim) throw FormatError(std::format("dim {} is outside [1, {}]", dim, max_dim));
        }

        [[nodiscard]] inline std::uint64_t norm_limit(const std::size_t dim){
            return static_cast<std::uint64_t>(int8_limit * int8_limit) * dim;
        }

        inline void check_entries(const Embedding& record){
            if (std::ranges::find(record, std::int8_t{-int8_limit - 1}) != record.end())
                throw FormatError(std::format("int8 entries must lie in [-{0}, {0}]", int8_limit));
        }

        [[nodiscard]] inline bool is_id_byte(const char c){
            const auto byte = static_cast<unsigned char>(c);
            return byte > 0x20 && byte != 0x7f;
        }

        inline void check_ids(const std::vector<std::string>& ids){
            if (ids.empty() || ids.size() > std::numeric_limits<std::uint32_t>::max())
                throw FormatError(std::format("a table needs between 1 and 2^32 - 1 ids, got {}", ids.size()));
            std::unordered_set<std::string_view> seen;
            for (std::size_t i = 0; i < ids.size(); ++i){
                const auto& id = ids[i];
                if (id.empty() || id.size() > std::numeric_limits<std::uint16_t>::max() || !std::ranges::all_of(id, is_id_byte))
                    throw FormatError(std::format("id {} must be 1 to 65535 printable non-space bytes", i));
                if (!seen.insert(id).second) throw FormatError(std::format("id {} repeats an earlier id", i));
            }
        }

        inline void check_shard(const Shard shard){
            if (shard.index >= shard.count)
                throw FormatError(std::format("shard {}/{} needs 0 <= index < count", shard.index, shard.count));
        }

        [[nodiscard]] inline std::pair<std::size_t, std::size_t> shard_range(const std::size_t size, const Shard shard){
            return {shard.index * size / shard.count, (shard.index + 1) * size / shard.count};
        }

        class Reader{
        public:
            explicit Reader(const rbp::ByteView bytes) : bytes_(bytes){}

            [[nodiscard]] rbp::ByteView take(const std::size_t size){
                if (size > bytes_.size()) throw FormatError("file is truncated");
                const auto taken = bytes_.first(size);
                bytes_ = bytes_.subspan(size);
                return taken;
            }

            template <std::unsigned_integral U>
            [[nodiscard]] U get(){
                const auto bytes = take(sizeof(U));
                U value = 0;
                for (std::size_t i = 0; i < sizeof(U); ++i) value |= static_cast<U>(static_cast<U>(bytes[i]) << (8 * i));
                return value;
            }

            [[nodiscard]] std::size_t preamble(const std::string_view magic){
                if (!std::ranges::equal(take(magic.size()), magic, {}, {}, [](const char c){ return static_cast<std::uint8_t>(c); }))
                    throw FormatError(std::format("expected a {} file", magic));
                if (const auto version = get<std::uint32_t>(); version != format_version)
                    throw FormatError(std::format("unsupported format version {}", version));
                const std::size_t dim = get<std::uint32_t>();
                check_dim(dim);
                return dim;
            }

            [[nodiscard]] std::size_t remaining() const{ return bytes_.size(); }

        private:
            rbp::ByteView bytes_;
        };

        template <class Record>
        struct PointCodec{
            using Point = typename decltype(Record::r)::value_type;

            static constexpr bool stores_norm = true;

            [[nodiscard]] static std::size_t width(){
                static const auto width = Point::generator().to_bytes().size();
                return width;
            }

            [[nodiscard]] static std::size_t record_size(const std::size_t dim){
                return (IPFE::OPT::b_size + dim) * width();
            }

            static void write(rbp::Bytes& out, const Record& record){
                for (const auto* points : {&record.r, &record.vec}){
                    for (const auto& point : *points){
                        if (point.is_identity()) out.insert(out.end(), width(), 0);
                        else{
                            const auto bytes = point.to_bytes();
                            out.insert(out.end(), bytes.begin(), bytes.end());
                        }
                    }
                }
            }

            [[nodiscard]] static Record read(const rbp::ByteView bytes, const std::size_t dim){
                Reader reader(bytes);
                return {read_points(reader, IPFE::OPT::b_size), read_points(reader, dim)};
            }

            [[nodiscard]] static std::vector<Point> read_points(Reader& reader, const std::size_t count){
                std::vector<Point> points;
                points.reserve(count);
                for (std::size_t i = 0; i < count; ++i) points.push_back(read_point(reader.take(width())));
                return points;
            }

            [[nodiscard]] static Point read_point(const rbp::ByteView slot){
                if (slot.front() != 0) return Point::from_bytes(slot);
                if (std::ranges::any_of(slot, [](const std::uint8_t byte){ return byte != 0; }))
                    throw FormatError("an identity point slot must be all zero bytes");
                return Point();
            }
        };
    }

    template <>
    struct Codec<Embedding>{
        static constexpr std::string_view magic = "PRVC";
        static constexpr bool stores_norm = false;

        [[nodiscard]] static std::size_t record_size(const std::size_t dim){
            return dim;
        }

        static void write(rbp::Bytes& out, const Embedding& record){
            for (const auto entry : record) out.push_back(static_cast<std::uint8_t>(entry));
        }

        [[nodiscard]] static Embedding read(const rbp::ByteView bytes, const std::size_t){
            Embedding record(bytes.size());
            std::ranges::copy(bytes, record.begin());
            detail::check_entries(record);
            return record;
        }
    };

    template <>
    struct Codec<Ct> : detail::PointCodec<Ct>{
        static constexpr std::string_view magic = "PRCT";
    };

    template <>
    struct Codec<Sk> : detail::PointCodec<Sk>{
        static constexpr std::string_view magic = "PRSK";
    };

    [[nodiscard]] inline std::uint64_t norm_sq(const Embedding& record){
        std::uint64_t total = 0;
        for (const std::int64_t entry : record) total += static_cast<std::uint64_t>(entry * entry);
        return total;
    }

    [[nodiscard]] inline Table<Embedding> make_vectors(std::vector<std::string> ids, std::vector<Embedding> records){
        detail::check_ids(ids);
        if (records.size() != ids.size())
            throw FormatError(std::format("{} ids need as many records, got {}", ids.size(), records.size()));
        const auto dim = records.front().size();
        detail::check_dim(dim);
        std::uint64_t max_norm_sq = 0;
        for (const auto& record : records){
            if (record.size() != dim) throw FormatError(std::format("every vector needs dim {}", dim));
            detail::check_entries(record);
            max_norm_sq = std::max(max_norm_sq, norm_sq(record));
        }
        return {std::move(ids), dim, max_norm_sq, std::move(records)};
    }

    [[nodiscard]] inline Table<Embedding> first(const Table<Embedding>& table, const std::size_t n){
        const auto count = static_cast<std::ptrdiff_t>(std::min(n, table.ids.size()));
        return make_vectors(
            {table.ids.begin(), table.ids.begin() + count}, {table.records.begin(), table.records.begin() + count}
        );
    }

    template <class F>
    [[nodiscard]] auto map_records(const Table<Embedding>& table, F f) -> Table<std::invoke_result_t<F, const Embedding&>>{
        Table<std::invoke_result_t<F, const Embedding&>> mapped{table.ids, table.dim, table.max_norm_sq, {}};
        mapped.records.reserve(table.records.size());
        for (const auto& record : table.records) mapped.records.push_back(f(record));
        return mapped;
    }

    [[nodiscard]] inline IPFE::IntVec widen(const Embedding& record){
        return {record.begin(), record.end()};
    }

    template <class Record>
    [[nodiscard]] rbp::Bytes encode_table(const Table<Record>& table){
        using Format = Codec<Record>;
        detail::check_dim(table.dim);
        detail::check_ids(table.ids);
        if (table.records.size() != table.ids.size()) throw FormatError("a table needs one record per id");
        rbp::Bytes out;
        detail::put_preamble(out, Format::magic, table.dim);
        detail::put(out, static_cast<std::uint32_t>(table.ids.size()));
        if constexpr (Format::stores_norm) detail::put(out, table.max_norm_sq);
        for (const auto& id : table.ids){
            detail::put(out, static_cast<std::uint16_t>(id.size()));
            out.insert(out.end(), id.begin(), id.end());
        }
        const auto size = Format::record_size(table.dim);
        for (const auto& record : table.records){
            const auto start = out.size();
            Format::write(out, record);
            if (out.size() - start != size) throw FormatError(std::format("a record does not have dim {}", table.dim));
        }
        return out;
    }

    template <class Record>
    [[nodiscard]] Table<Record> decode_table(const rbp::ByteView bytes, const Shard shard = {}){
        using Format = Codec<Record>;
        detail::check_shard(shard);
        detail::Reader reader(bytes);
        const auto dim = reader.preamble(Format::magic);
        const std::size_t count = reader.get<std::uint32_t>();
        std::uint64_t max_norm_sq = 0;
        if constexpr (Format::stores_norm){
            max_norm_sq = reader.get<std::uint64_t>();
            if (max_norm_sq > detail::norm_limit(dim))
                throw FormatError(std::format("max_norm_sq {} exceeds {} for dim {}", max_norm_sq, detail::norm_limit(dim), dim));
        }
        std::vector<std::string> ids;
        for (std::size_t i = 0; i < count; ++i){
            const auto id = reader.take(reader.get<std::uint16_t>());
            ids.emplace_back(id.begin(), id.end());
        }
        detail::check_ids(ids);
        const auto size = Format::record_size(dim);
        if (reader.remaining() != count * size)
            throw FormatError(std::format("{} records of {} bytes need {} bytes, found {}", count, size, count * size, reader.remaining()));
        const auto records = reader.take(count * size);
        const auto record = [&](const std::size_t i){ return Format::read(records.subspan(i * size, size), dim); };
        const auto [begin, end] = detail::shard_range(count, shard);
        Table<Record> table{
            {ids.begin() + static_cast<std::ptrdiff_t>(begin), ids.begin() + static_cast<std::ptrdiff_t>(end)}, dim, max_norm_sq, {}
        };
        table.records.reserve(end - begin);
        if constexpr (Format::stores_norm){
            for (auto i = begin; i < end; ++i) table.records.push_back(record(i));
        }
        else{
            for (std::size_t i = 0; i < count; ++i){
                auto decoded = record(i);
                table.max_norm_sq = std::max(table.max_norm_sq, norm_sq(decoded));
                if (i >= begin && i < end) table.records.push_back(std::move(decoded));
            }
        }
        return table;
    }

    template <std::integral T>
    [[nodiscard]] std::optional<T> parse_integer(const std::string_view text){
        T value = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (error != std::errc{} || end != text.data() + text.size()) return std::nullopt;
        return value;
    }

    [[nodiscard]] inline Shard parse_shard(const std::string_view text){
        const auto slash = text.find('/');
        const auto index = parse_integer<std::size_t>(text.substr(0, slash));
        const auto count = slash == std::string_view::npos ? std::nullopt : parse_integer<std::size_t>(text.substr(slash + 1));
        if (!index || !count || *index >= *count) throw FormatError(std::format("shard '{}' must be i/N with 0 <= i < N", text));
        return {*index, *count};
    }

    [[nodiscard]] inline rbp::Bytes encode_msk(const Msk& msk){
        rbp::Bytes out;
        detail::put_preamble(out, detail::msk_magic, msk.a.cols());
        for (const auto* matrix : {&msk.a, &msk.b, &msk.bi}){
            for (std::size_t i = 0; i < matrix->rows(); ++i){
                for (std::size_t j = 0; j < matrix->cols(); ++j){
                    const auto bytes = matrix->at(i, j).to_bytes();
                    out.insert(out.end(), bytes.begin(), bytes.end());
                }
            }
        }
        return out;
    }

    [[nodiscard]] inline Msk decode_msk(const rbp::ByteView bytes){
        constexpr auto b_size = IPFE::OPT::b_size;
        detail::Reader reader(bytes);
        const auto dim = reader.preamble(detail::msk_magic);
        const auto width = rbp::Zp<Curve>::byte_size();
        if (reader.remaining() != (2 * dim + 2 * b_size * b_size) * width)
            throw FormatError(std::format("a master key of dim {} has the wrong length", dim));
        const auto matrix = [&](const std::size_t rows, const std::size_t cols){
            rbp::Matrix<Curve> m(rows, cols);
            for (std::size_t i = 0; i < rows; ++i)
                for (std::size_t j = 0; j < cols; ++j) m.at(i, j) = rbp::Zp<Curve>::from_bytes(reader.take(width));
            return m;
        };
        return {matrix(2, dim), matrix(b_size, b_size), matrix(b_size, b_size)};
    }

    [[nodiscard]] inline rbp::Bytes read_file(const std::filesystem::path& path){
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::filesystem::filesystem_error("cannot open for reading", path, std::error_code(errno, std::generic_category()));
        rbp::Bytes bytes(std::filesystem::file_size(path));
        if (!in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
            throw std::filesystem::filesystem_error("cannot read", path, std::make_error_code(std::errc::io_error));
        return bytes;
    }

    inline void write_file(const std::filesystem::path& path, const std::string_view text){
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) throw std::filesystem::filesystem_error("cannot open for writing", path, std::error_code(errno, std::generic_category()));
        if (!out.write(text.data(), static_cast<std::streamsize>(text.size())).flush())
            throw std::filesystem::filesystem_error("cannot write", path, std::make_error_code(std::errc::io_error));
    }

    inline void write_file(const std::filesystem::path& path, const rbp::ByteView bytes){
        write_file(path, std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    }
}

#endif
