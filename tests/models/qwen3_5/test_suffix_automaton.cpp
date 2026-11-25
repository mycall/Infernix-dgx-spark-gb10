#include "models/qwen3_5/program/suffix_automaton.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using infernix::models::qwen3_5::detail::DualGenSuffixAutomaton;
using infernix::models::qwen3_5::detail::SuffixAutomaton;

static void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

int main() {
    try {
        // 1. Empty and single-token
        SuffixAutomaton empty_sa;
        require(!empty_sa.lookup().has_value(), "empty automaton has no lookup match");
        empty_sa.append(1);
        require(!empty_sa.lookup().has_value(), "single token has no lookup match");

        // 2. Simple repetition (test from sa_spec: test_suffix_automaton.cc)
        SuffixAutomaton rep_sa;
        rep_sa.append(1);
        rep_sa.append(1);
        rep_sa.append(1);
        auto res = rep_sa.lookup();
        require(res.has_value(), "repetition should find lookup match");
        require(res->len == 2, "repetition match len should be 2");
        require(res->pos == 2, "repetition match pos should be 2");

        rep_sa.append(1);
        res = rep_sa.lookup();
        require(res.has_value(), "repetition 4 tokens should find match");
        require(res->len == 3, "repetition match len should be 3");
        require(res->pos == 3, "repetition match pos should be 3");

        // 3. Alternating repetition (test from sa_spec)
        SuffixAutomaton alt_sa;
        alt_sa.append(1);
        alt_sa.append(2);
        require(!alt_sa.lookup().has_value(), "distinct initial tokens have no match");
        alt_sa.append(1);
        res = alt_sa.lookup();
        require(res.has_value() && res->len == 1 && res->pos == 1, "first repeat of token 1");
        alt_sa.append(2);
        res = alt_sa.lookup();
        require(res.has_value() && res->len == 2 && res->pos == 2, "repeat of sequence [1, 2]");

        // 4. Overlapping repeats ("ababa" -> [1, 2, 1, 2, 1])
        SuffixAutomaton ababa;
        for (int t : {1, 2, 1, 2, 1}) { ababa.append(t); }
        res = ababa.lookup();
        require(res.has_value() && res->len == 3 && res->pos == 3, "ababa has suffix 'aba' of len 3");

        // 5. Baseten test_add_request pattern from sa_spec test_api.py
        SuffixAutomaton api_sa;
        for (int t : {1, 2, 3, 4, 1, 2, 3, 1, 2}) { api_sa.append(t); }
        api_sa.append(3);
        api_sa.append(4);
        res = api_sa.lookup();
        require(res.has_value(), "api pattern should match");
        require(res->len == 4, "api pattern match length should be 4");
        require(res->pos == 4, "api pattern continuation pos should be 4");

        // Propose from this match
        auto match = api_sa.propose(std::vector<int>{1, 2, 3, 4}, 2, 4);
        require(match.matched == 4, "matched 4 tokens");
        require(match.tokens == std::vector<int>({1, 2}), "draft tokens should be [1, 2]");

        // 6. Test CUDA graph pattern from sa_spec (prompt + new_tokens)
        SuffixAutomaton graph_sa;
        for (int t : {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1}) { graph_sa.append(t); }
        graph_sa.append(2);
        graph_sa.append(3);
        res = graph_sa.lookup();
        // SAM correctly discovers the maximal 8-token matching suffix [1, 2, 3, 4, 5, 1, 2, 3]
        require(res.has_value() && res->len == 8, "graph pattern should match maximal suffix (8 tokens)");
        auto graph_match = graph_sa.propose(std::vector<int>{1, 2, 3}, 2, 3);
        require(graph_match.tokens == std::vector<int>({4, 5}), "drafts should be [4, 5]");

        // 7. Boundary test: ensure proposals never cross boundary markers
        SuffixAutomaton b_sa;
        b_sa.set_boundaries({99});
        b_sa.ingest(std::vector<int>{10, 20, 30, 40, 99, 50, 60});
        auto b_match = b_sa.propose(std::vector<int>{10, 20}, 5, 2);
        require(b_match.matched == 2, "b_match length 2");
        require(b_match.tokens == std::vector<int>({30, 40}), "proposal stops before boundary 99");

        // 8. Round admission and lookahead checks
        SuffixAutomaton round_sa(256);
        std::vector<int> src(128);
        std::iota(src.begin(), src.end(), 100);
        round_sa.ingest(src);
        for (unsigned n : {4, 8, 12, 16, 32, 64}) {
            const auto m = round_sa.propose(std::span<const int>(src).first(n), 15, 4);
            require(m.matched == n, "exact match length");
            require(m.tokens == std::vector<int>(src.begin() + n, src.begin() + n + 15), "exact span continuation");
        }

        // 9. Boundary-adjacent state must walk suffix link to find a valid continuation
        SuffixAutomaton link_sa;
        link_sa.ingest(std::vector<int>{2, 3, 4, 5, 99, 100});
        link_sa.ingest(std::vector<int>{1, 2, 3, 4, 5});
        const auto link_match = link_sa.propose(std::vector<int>{1, 2, 3, 4, 5}, 5, 4);
        require(link_match.matched == 4, "boundary-adjacent state walks suffix link to len 4");
        require(link_match.tokens == std::vector<int>({99, 100}),
                "boundary-adjacent state returns continuation from suffix link");

        // 10. Recency preference: more recent occurrence supersedes older occurrence
        SuffixAutomaton recency_sa;
        recency_sa.ingest(std::vector<int>{10, 20, 30, 40, 100, 101, 102});
        recency_sa.ingest(std::vector<int>{10, 20, 30, 40, 200, 201, 202, 203});
        const auto rec_match = recency_sa.propose(std::vector<int>{10, 20, 30, 40}, 10, 4);
        require(rec_match.matched == 4, "recency match len 4");
        require(rec_match.tokens == std::vector<int>({200, 201, 202, 203}),
                "recency prefers the most recent occurrence");

        // 11. Boundary-ended latest occurrence falls back cleanly to earlier valid occurrence
        SuffixAutomaton valid_fb_sa;
        valid_fb_sa.ingest(std::vector<int>{10, 20, 30, 40, 300, 301, 302});
        valid_fb_sa.ingest(std::vector<int>{10, 20, 30, 40});
        const auto vfb_match = valid_fb_sa.propose(std::vector<int>{10, 20, 30, 40}, 10, 4);
        require(vfb_match.matched == 4, "valid fallback match len 4");
        require(vfb_match.tokens == std::vector<int>({300, 301, 302}),
                "boundary-ended latest occurrence falls back to earlier valid continuation");

        // 12. Eviction resilience: older evicted occurrence does not block newer live occurrence
        SuffixAutomaton evict_sa(64);
        evict_sa.ingest(std::vector<int>{10, 20, 30, 40, 111, 112});
        for (int i = 0; i < 60; ++i) { evict_sa.append(500 + i); }
        evict_sa.ingest(std::vector<int>{10, 20, 30, 40, 222, 223});
        const auto ev_match = evict_sa.propose(std::vector<int>{10, 20, 30, 40}, 10, 4);
        require(ev_match.matched == 4, "eviction resilient match len 4");
        require(ev_match.tokens == std::vector<int>({222, 223}),
                "newer live occurrence still matches after oldest aged out");

        // 13. Occurrence count and distance metadata
        require(rec_match.occurrences >= 2, "occurrence count tracks repeats");
        require(ev_match.distance <= 64, "distance tracks distance from corpus tip");

        // 14. Compaction bounds storage under >4x token_capacity appends
        SuffixAutomaton compact_sa(64);
        for (int i = 0; i < 300; ++i) { compact_sa.append(1000 + i); }
        require(compact_sa.size() < 2 * 64 || compact_sa.size() <= 1024, "compaction bounded size");
        require(compact_sa.node_count() <= 4 * 1024, "compaction bounded nodes");

        // 15. Cross-request isolation & longest match across separate ingest calls
        SuffixAutomaton multi_req_sa(256);
        multi_req_sa.ingest(std::vector<int>{401, 402, 403, 404, 405, 406, 407, 408});
        multi_req_sa.ingest(std::vector<int>{501, 502, 503, 504, 505, 506, 507, 508});
        const auto m_r1 = multi_req_sa.propose(std::vector<int>{401, 402, 403, 404, 405, 406}, 8, 6);
        require(m_r1.matched == 6 && m_r1.tokens == std::vector<int>({407, 408}),
                "multi-request ingest recovers first request tail and stops at request boundary");
        const auto m_cross_bleed = multi_req_sa.propose(std::vector<int>{406, 407, 408, 501, 502, 503}, 4, 6);
        require(m_cross_bleed.tokens.empty(), "ingest boundary prevents cross-request concatenation match");

        // 16. Frequency-weighted continuation beats a single recent outlier, while ties keep recency
        SuffixAutomaton freq_sa(256);
        for (int rep = 0; rep < 3; ++rep) {
            freq_sa.ingest(std::vector<int>{11, 22, 33, 44, 701, 702, 703});
        }
        freq_sa.ingest(std::vector<int>{11, 22, 33, 44, 801, 802, 803});
        const auto freq_match = freq_sa.propose(std::vector<int>{11, 22, 33, 44}, 8, 4);
        require(freq_match.matched == 4 && freq_match.tokens == std::vector<int>({701, 702, 703}),
                "frequency-weighted selection prefers 3x pattern over single recent outlier");

        SuffixAutomaton tie_sa(256);
        tie_sa.ingest(std::vector<int>{51, 52, 53, 54, 901, 902});
        tie_sa.ingest(std::vector<int>{51, 52, 53, 54, 911, 912});
        const auto tie_match = tie_sa.propose(std::vector<int>{51, 52, 53, 54}, 8, 4);
        require(tie_match.matched == 4 && tie_match.tokens == std::vector<int>({911, 912}),
                "equal-frequency continuations still break ties by recency");

        // 17. 2-branch alt_token / alt_occurrences extraction
        SuffixAutomaton branch_sa(256);
        for (int rep = 0; rep < 3; ++rep) {
            branch_sa.ingest(std::vector<int>{21, 22, 23, 24, 301, 302, 303});
        }
        for (int rep = 0; rep < 2; ++rep) {
            branch_sa.ingest(std::vector<int>{21, 22, 23, 24, 401, 402, 403});
        }
        const auto br_match = branch_sa.propose(std::vector<int>{21, 22, 23, 24}, 8, 4);
        require(br_match.matched == 4 && br_match.tokens == std::vector<int>({301, 302, 303}),
                "2-branch primary selects 3x pattern");
        require(br_match.alt_token == 401 && br_match.alt_occurrences == 2,
                "2-branch alt_token and alt_occurrences expose runner-up continuation");

        // 18. O(1) incremental Cursor matches propose(history) and invalidates on clear()
        SuffixAutomaton cur_sa(256);
        cur_sa.ingest(std::vector<int>{60, 61, 62, 63, 64, 65, 66, 67, 68, 69});
        auto cur = cur_sa.cursor_from(std::span<const int>{});
        const std::vector<int> stream{999, 60, 61, 62, 63, 64};
        for (std::size_t idx = 0; idx < stream.size(); ++idx) {
            cur_sa.advance_cursor(cur, stream[idx]);
            const std::span<const int> pref(stream.data(), idx + 1);
            const auto from_cur  = cur_sa.propose_from_cursor(cur, 8, 4);
            const auto from_hist = cur_sa.propose(pref, 8, 4);
            require(from_cur.matched == from_hist.matched && from_cur.tokens == from_hist.tokens &&
                        from_cur.alt_token == from_hist.alt_token &&
                        from_cur.alt_occurrences == from_hist.alt_occurrences,
                    "O(1) cursor matches propose(history) bit-for-bit");
        }
        const std::uint64_t old_gen = cur.generation;
        cur_sa.clear();
        require(cur_sa.generation() > old_gen, "clear() bumps SAM generation");
        require(cur_sa.propose_from_cursor(cur, 8, 4).tokens.empty(),
                "stale cursor returns empty after clear()/generation bump");

        // 19. DualGenSuffixAutomaton rotation preserves prior-bank patterns & eliminates rebuild stalls
        DualGenSuffixAutomaton dg_sa(512); // bank capacity 256 each
        dg_sa.ingest(std::vector<int>{10, 11, 12, 13, 14, 15, 16, 17});
        for (int r = 0; r < 10; ++r) { // ~120 tokens -> active bank still, then second batch rotates cleanly
            std::vector<int> blk(20);
            std::iota(blk.begin(), blk.end(), 2000 + r * 30);
            dg_sa.ingest(blk);
        }
        // Recent prior-bank pattern survives rotation into standby_
        std::vector<int> pinned{77, 78, 79, 80, 81, 82, 83, 84, 85, 86};
        dg_sa.ingest(pinned);
        const std::uint64_t gen_before = dg_sa.generation();
        for (int r = 0; r < 10; ++r) { // overflow active_ once so pinned moves into standby_ without double-rotating
            std::vector<int> fill(16);
            std::iota(fill.begin(), fill.end(), 5000 + r * 20);
            dg_sa.ingest(fill);
        }
        require(dg_sa.generation() > gen_before, "DualGen rotated cleanly into standby bank");
        const auto standby_hit = dg_sa.propose(std::vector<int>{77, 78, 79, 80, 81, 82}, 8, 6);
        require(standby_hit.matched == 6 && standby_hit.tokens == std::vector<int>({83, 84, 85, 86}),
                "DualGen serves matches from standby bank seamlessly after rotation");

        // Item 3 Benchmark: Single-SAM (compact_if_needed) vs DualGenSuffixAutomaton over 1,000,000 tokens
        constexpr std::size_t kBenchCap   = 262144;
        constexpr int kNumReqs            = 2000;
        constexpr int kReqLen             = 500;
        std::vector<std::vector<int>> requests(static_cast<std::size_t>(kNumReqs),
                                               std::vector<int>(static_cast<std::size_t>(kReqLen)));
        std::mt19937 rng(42);
        std::vector<int> repeated_template(32);
        std::iota(repeated_template.begin(), repeated_template.end(), 100);
        for (int r = 0; r < kNumReqs; ++r) {
            for (int t = 0; t < kReqLen; ++t) {
                requests[static_cast<std::size_t>(r)][static_cast<std::size_t>(t)] =
                    (t < 32) ? repeated_template[static_cast<std::size_t>(t)]
                             : static_cast<int>(1000 + (rng() % 50000));
            }
        }

        const auto bench_ingest = [&](auto& sam, const char* label) {
            std::vector<double> lat_us;
            lat_us.reserve(static_cast<std::size_t>(kNumReqs));
            std::size_t hits = 0;
            const auto t0    = std::chrono::steady_clock::now();
            for (int r = 0; r < kNumReqs; ++r) {
                const auto s0 = std::chrono::steady_clock::now();
                sam.ingest(requests[static_cast<std::size_t>(r)]);
                const auto s1 = std::chrono::steady_clock::now();
                lat_us.push_back(std::chrono::duration<double, std::micro>(s1 - s0).count());
                if ((r % 20) == 19) {
                    const auto m = sam.propose(std::span<const int>(repeated_template).first(8), 16, 8);
                    if (!m.tokens.empty()) { ++hits; }
                }
            }
            const auto t1         = std::chrono::steady_clock::now();
            const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::sort(lat_us.begin(), lat_us.end());
            const double p50_us       = lat_us[lat_us.size() / 2];
            const double p99_us       = lat_us[(lat_us.size() * 99) / 100];
            const double max_stall_ms = lat_us.back() / 1000.0;
            std::cout << std::fixed << std::setprecision(3) << "[SAM Bench] " << label
                      << ": total_ms=" << total_ms << " p50_us=" << p50_us << " p99_us=" << p99_us
                      << " max_stall_ms=" << max_stall_ms << " hits=" << hits << '\n';
            return max_stall_ms;
        };

        SuffixAutomaton single_sam(kBenchCap, /*compact_enabled=*/true);
        DualGenSuffixAutomaton dual_sam(kBenchCap);
        const double single_max_ms = bench_ingest(single_sam, "Single-SAM compact_if_needed");
        const double dual_max_ms   = bench_ingest(dual_sam, "DualGenSuffixAutomaton");
        require(dual_max_ms < single_max_ms,
                "DualGenSuffixAutomaton max ingest stall must be lower than stop-the-world compaction");

        std::cout << "All Suffix Automaton unit tests passed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << '\n';
        return 1;
    }
}
