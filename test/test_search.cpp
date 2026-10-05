#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <privret/search.hpp>

using namespace privret;

namespace{
    std::vector<Embedding> random_embeddings(std::mt19937_64& engine, const std::size_t count, const std::size_t dim){
        std::uniform_int_distribution<int> entry(-127, 127);
        std::vector<Embedding> records(count, Embedding(dim));
        for (auto& record : records) for (auto& e : record) e = static_cast<std::int8_t>(entry(engine));
        return records;
    }

    std::vector<std::string> hit_docs(const Ranking& ranking){
        std::vector<std::string> docs;
        for (const auto& hit : ranking.hits) docs.push_back(hit.doc);
        return docs;
    }
}

TEST(RankingTest, OrdersByScoreThenDocId){
    const std::vector<Hit> hits{{"b", 5}, {"a", 5}, {"c", 9}, {"d", -1}};

    EXPECT_EQ(top_k(hits, 3), (std::vector<Hit>{{"c", 9}, {"a", 5}, {"b", 5}}));
    EXPECT_EQ(top_k(hits, 10), (std::vector<Hit>{{"c", 9}, {"a", 5}, {"b", 5}, {"d", -1}}));
}

TEST(IsqrtTest, IsExactAtTheBoundary){
    EXPECT_EQ(isqrt(0), 0);
    EXPECT_EQ(isqrt(6), 2);
    EXPECT_EQ(isqrt(36), 6);
    EXPECT_EQ(isqrt(35), 5);
    EXPECT_EQ(isqrt(96774ull * 96774), 96774);
    EXPECT_EQ(isqrt((16129ull * 4096) * (16129ull * 4096)), 16129ull * 4096);
}

TEST(RunTextTest, RoundTripsAndRejectsMalformedLines){
    const privret::Run run{{"q1", {{"d3", 12}, {"d1", -4}}}, {"q2", {{"d1", 0}}}};
    const auto text = format_run(run);
    EXPECT_EQ(text, "q1 Q0 d3 1 12 privret\nq1 Q0 d1 2 -4 privret\nq2 Q0 d1 1 0 privret\n");
    EXPECT_EQ(parse_run(text), run);

    for (const auto line : {"q1 Q0 d1 1 x privret", "q1 Q0 d1 1 5", "q1 Q1 d1 1 5 privret"}){
        SCOPED_TRACE(line);
        EXPECT_THROW((void)parse_run(line), FormatError);
    }
}

TEST(MergeTest, RejectsDuplicateDocs){
    const privret::Run run{{"q1", {{"d1", 3}, {"d2", 1}}}};

    EXPECT_THROW((void)merge({run, run}, 10), FormatError);
}

TEST(EncryptedSearchTest, MatchesPlainSearchExactly){
    constexpr std::size_t dim = 6;
    std::mt19937_64 engine(7);
    auto random_docs = random_embeddings(engine, 8, dim);
    const auto twin = random_docs.back();
    random_docs.front() = Embedding(dim, 127);
    const auto docs = make_vectors(
        {"d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d10"},
        {random_docs[0], random_docs[1], twin, random_docs[2], random_docs[3], random_docs[4], random_docs[5],
         random_docs[6], random_docs[7], twin}
    );
    auto random_queries = random_embeddings(engine, 3, dim);
    const auto queries = make_vectors(
        {"q0", "q1", "q2", "q3", "q4"},
        {Embedding(dim, -127), twin, random_queries[0], random_queries[1], random_queries[2]}
    );

    const auto msk = IPFE::OPT::setup<Curve>(dim);
    const auto corpus_bytes = encode_table(map_records(docs, [&](const Embedding& v){ return IPFE::OPT::enc(msk, widen(v)); }));
    const auto key_bytes = encode_table(map_records(queries, [&](const Embedding& v){ return IPFE::OPT::keygen(msk, widen(v)); }));
    const auto corpus = decode_table<Ct>(corpus_bytes);
    const auto keys = decode_table<Sk>(key_bytes);
    const auto table = score_table({.docs = corpus, .queries = keys, .top_k = 10});

    const auto plain = search_plain({.docs = docs, .queries = queries, .top_k = 10});
    EXPECT_EQ(plain.front().hits.back(), (Hit{"d0", -96774}));
    const auto ranked = hit_docs(plain[1]);
    const auto tie = std::ranges::find(ranked, "d10");
    ASSERT_NE(tie, ranked.end());
    EXPECT_EQ(*(tie + 1), "d2");
    EXPECT_EQ(search_encrypted({.docs = corpus, .queries = keys, .top_k = 10}, table), plain);

    const auto plain_top3 = search_plain({.docs = docs, .queries = queries, .top_k = 3});
    EXPECT_EQ(search_encrypted({.docs = corpus, .queries = keys, .top_k = 3}, table), plain_top3);

    std::vector<privret::Run> shard_runs;
    for (std::size_t i = 0; i < 3; ++i){
        const auto shard = decode_table<Ct>(corpus_bytes, {i, 3});
        shard_runs.push_back(search_encrypted({.docs = shard, .queries = keys, .top_k = 3}, table));
    }
    EXPECT_EQ(merge(shard_runs, 3), plain_top3);

    const auto other_msk = IPFE::OPT::setup<Curve>(dim);
    const auto foreign_keys = map_records(queries, [&](const Embedding& v){ return IPFE::OPT::keygen(other_msk, widen(v)); });
    EXPECT_THROW((void)search_encrypted({.docs = corpus, .queries = foreign_keys, .top_k = 3}, table), DecryptError);

    const auto short_msk = IPFE::OPT::setup<Curve>(dim - 1);
    const auto short_keys = map_records(
        make_vectors({"s0"}, {Embedding(dim - 1, 1)}), [&](const Embedding& v){ return IPFE::OPT::keygen(short_msk, widen(v)); }
    );
    EXPECT_THROW((void)search_encrypted({.docs = corpus, .queries = short_keys, .top_k = 3}, table), FormatError);
}
