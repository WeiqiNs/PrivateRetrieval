#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <privret/format.hpp>

using namespace privret;

namespace{
    Table<Embedding> sample_vectors(){
        return make_vectors({"a", "b"}, {{3, 4, 0}, {-127, 0, 1}});
    }

    Table<Embedding> counting_vectors(const std::size_t count){
        std::vector<std::string> ids;
        std::vector<Embedding> records;
        for (std::size_t i = 0; i < count; ++i){
            ids.push_back("d" + std::to_string(i));
            records.push_back({static_cast<std::int8_t>(i)});
        }
        return make_vectors(std::move(ids), std::move(records));
    }

    template <class Record>
    void expect_same_points(const std::vector<Record>& actual, const std::vector<Record>& expected){
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i){
            EXPECT_EQ(actual[i].r, expected[i].r);
            EXPECT_EQ(actual[i].vec, expected[i].vec);
        }
    }

    void put_u64(rbp::Bytes& bytes, const std::size_t offset, const std::uint64_t value){
        for (std::size_t i = 0; i < 8; ++i) bytes[offset + i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
}

TEST(FormatTest, RoundTripsVectorsAndDerivesTheNorm){
    const auto vectors = sample_vectors();
    EXPECT_EQ(vectors.dim, 3);
    EXPECT_EQ(vectors.max_norm_sq, 16130);

    const auto bytes = encode_table(vectors);
    EXPECT_EQ(bytes, (rbp::Bytes{
        'P', 'R', 'V', 'C', 1, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0,
        1, 0, 'a', 1, 0, 'b',
        3, 4, 0, 0x81, 0, 1
    }));

    const auto decoded = decode_table<Embedding>(bytes);
    EXPECT_EQ(decoded.ids, vectors.ids);
    EXPECT_EQ(decoded.dim, 3);
    EXPECT_EQ(decoded.max_norm_sq, 16130);
    EXPECT_EQ(decoded.records, vectors.records);
}

TEST(FormatTest, RejectsMalformedVectors){
    const auto valid = encode_table(sample_vectors());
    const std::vector<std::pair<std::string_view, std::function<void(rbp::Bytes&)>>> mutations{
        {"magic PRVX", [](rbp::Bytes& b){ b[3] = 'X'; }},
        {"version 2", [](rbp::Bytes& b){ b[4] = 2; }},
        {"dim 0", [](rbp::Bytes& b){ b[8] = 0; }},
        {"dim 4097", [](rbp::Bytes& b){ b[8] = 0x01, b[9] = 0x10; }},
        {"count 0", [](rbp::Bytes& b){ b[12] = 0; }},
        {"truncated by one byte", [](rbp::Bytes& b){ b.pop_back(); }},
        {"one trailing byte", [](rbp::Bytes& b){ b.push_back(0); }},
        {"entry -128", [](rbp::Bytes& b){ b[26] = 0x80; }},
        {"duplicate id", [](rbp::Bytes& b){ b[21] = 'a'; }},
        {"id with a space", [](rbp::Bytes& b){ b[21] = ' '; }},
        {"empty id", [](rbp::Bytes& b){ b[19] = 0; }},
    };
    for (const auto& [name, mutate] : mutations){
        SCOPED_TRACE(name);
        auto bytes = valid;
        mutate(bytes);
        EXPECT_THROW((void)decode_table<Embedding>(bytes), FormatError);
    }
    EXPECT_THROW((void)decode_table<Ct>(valid), FormatError);
}

TEST(FormatTest, RoundTripsCiphertextsKeysAndTheIdentityPoint){
    const auto msk = IPFE::OPT::setup<Curve>(3);
    const auto vectors = sample_vectors();
    const auto cts = map_records(vectors, [&](const Embedding& v){ return IPFE::OPT::enc(msk, widen(v)); });
    const auto sks = map_records(vectors, [&](const Embedding& v){ return IPFE::OPT::keygen(msk, widen(v)); });

    const auto ct_bytes = encode_table(cts);
    const auto sk_bytes = encode_table(sks);
    EXPECT_EQ(ct_bytes.size(), 24 + 6 + 2 * 7 * 49);
    EXPECT_EQ(sk_bytes.size(), 24 + 6 + 2 * 7 * 97);

    const auto decoded_cts = decode_table<Ct>(ct_bytes);
    EXPECT_EQ(decoded_cts.ids, vectors.ids);
    EXPECT_EQ(decoded_cts.max_norm_sq, 16130);
    expect_same_points(decoded_cts.records, cts.records);
    expect_same_points(decode_table<Sk>(sk_bytes).records, sks.records);

    auto with_identity = cts;
    with_identity.records[0].r[1] = rbp::G1<Curve>();
    const auto identity_bytes = encode_table(with_identity);
    EXPECT_EQ(identity_bytes.size(), ct_bytes.size());
    const auto decoded_identity = decode_table<Ct>(identity_bytes);
    EXPECT_TRUE(decoded_identity.records[0].r[1].is_identity());
    expect_same_points(decoded_identity.records, with_identity.records);

    auto at_limit = ct_bytes;
    put_u64(at_limit, 16, 127 * 127 * 3);
    EXPECT_EQ(decode_table<Ct>(at_limit).max_norm_sq, 127 * 127 * 3);
    auto over_limit = ct_bytes;
    put_u64(over_limit, 16, 127 * 127 * 3 + 1);
    EXPECT_THROW((void)decode_table<Ct>(over_limit), FormatError);

    auto corrupt = ct_bytes;
    corrupt[30 + 24] ^= 0xff;
    EXPECT_THROW((void)decode_table<Ct>(corrupt), rbp::DecodeError);
}

TEST(FormatTest, DecodesOnlyTheShardsRecords){
    struct Case{
        std::size_t docs;
        Shard shard;
        std::vector<std::string> ids;
    };
    const std::vector<Case> cases{
        {7, {0, 3}, {"d0", "d1"}},
        {7, {1, 3}, {"d2", "d3"}},
        {7, {2, 3}, {"d4", "d5", "d6"}},
        {2, {0, 3}, {}},
        {2, {1, 3}, {"d0"}},
        {2, {2, 3}, {"d1"}},
    };
    for (const auto& [docs, shard, ids] : cases){
        SCOPED_TRACE(std::to_string(shard.index) + "/" + std::to_string(shard.count) + " of " + std::to_string(docs));
        const auto whole = counting_vectors(docs);
        const auto part = decode_table<Embedding>(encode_table(whole), shard);
        EXPECT_EQ(part.ids, ids);
        EXPECT_EQ(part.dim, 1);
        EXPECT_EQ(part.max_norm_sq, whole.max_norm_sq);
        ASSERT_EQ(part.records.size(), ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i)
            EXPECT_EQ(part.records[i], Embedding{static_cast<std::int8_t>(std::stoi(ids[i].substr(1)))});
    }

    EXPECT_EQ(parse_shard("1/3"), (Shard{1, 3}));
    for (const auto text : {"3/3", "0/0", "1", "a/3", "1/3x"}){
        SCOPED_TRACE(text);
        EXPECT_THROW((void)parse_shard(text), FormatError);
    }
}

TEST(FormatTest, RoundTripsTheMasterKey){
    const auto msk = IPFE::OPT::setup<Curve>(3);
    const auto bytes = encode_msk(msk);
    EXPECT_EQ(bytes.size(), 12 + (2 * 3 + 2 * 16) * 32);

    const auto decoded = decode_msk(bytes);
    EXPECT_EQ(decoded.a, msk.a);
    EXPECT_EQ(decoded.b, msk.b);
    EXPECT_EQ(decoded.bi, msk.bi);

    auto truncated = bytes;
    truncated.pop_back();
    EXPECT_THROW((void)decode_msk(truncated), FormatError);

    auto out_of_range = bytes;
    std::fill_n(out_of_range.begin() + 12, 32, 0xff);
    EXPECT_THROW((void)decode_msk(out_of_range), rbp::DecodeError);
}
