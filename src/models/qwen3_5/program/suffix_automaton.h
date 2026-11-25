#pragma once

#include "models/qwen3_5/ngram.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernix::models::qwen3_5::detail {

// Recency-, frequency-, and continuation-aware Suffix Automaton (SAM) speculative draft proposer.
// Represents all substrings of the live window (bounded to token_capacity_ via tail compaction at
// 2 * token_capacity_) in linear space O(N) with at most 2N states and 3N transitions. Each state
// tracks its most recent occurrences (last_pos/prev_pos) and most recent occurrences with valid
// non-boundary continuations (valid_pos/prev_valid_pos) plus occurrence count (occ_count).
class SuffixAutomaton {
public:
    using Token = TokenId;
    using Match = NgramMatch;

    struct LookupResult {
        int32_t pos = 0;
        int32_t len = 0;
    };

    struct Cursor {
        int32_t state            = 0;
        int32_t matched          = 0;
        std::uint64_t generation = 0;
    };

    explicit SuffixAutomaton(std::size_t token_capacity = 1U << 20, bool compact_enabled = true)
        : token_capacity_(power_of_two(token_capacity) && token_capacity >= 64 ? token_capacity : 0),
          compact_enabled_(compact_enabled) {
        if (token_capacity_ == 0) {
            throw std::invalid_argument("invalid suffix automaton capacity");
        }
        nodes_.push_back(Node{});
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return token_capacity_; }
    [[nodiscard]] std::size_t size() const noexcept { return tokens_.size(); }
    [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }

    void clear() {
        nodes_.clear();
        edges_.clear();
        edge_table_.clear();
        edge_table_count_ = 0;
        tokens_.clear();
        nodes_.push_back(Node{});
        last_ = 0;
        ++generation_;
    }

    void boundary() {
        tokens_.push_back(-1);
        last_ = 0;
        compact_if_needed();
    }

    void set_boundaries(std::vector<Token> tokens) {
        boundaries_ = std::move(tokens);
        std::sort(boundaries_.begin(), boundaries_.end());
    }

    [[nodiscard]] bool is_boundary(Token t) const noexcept {
        if (t < 0) { return true; }
        return std::binary_search(boundaries_.begin(), boundaries_.end(), t);
    }

    void append(Token token) {
        if (is_boundary(token)) {
            boundary();
            return;
        }
        append_non_boundary(token);
        compact_if_needed();
    }

    void reserve_additional(std::size_t extra_tokens) {
        if (extra_tokens == 0) { return; }
        const std::size_t cap_tokens = std::min(tokens_.size() + extra_tokens, 2 * token_capacity_);
        if (cap_tokens > tokens_.capacity()) { tokens_.reserve(cap_tokens); }
        const std::size_t cap_nodes = std::min(nodes_.size() + 2 * extra_tokens, 4 * token_capacity_);
        if (cap_nodes > nodes_.capacity()) { nodes_.reserve(cap_nodes); }
        const std::size_t cap_edges = std::min(edges_.size() + extra_tokens, 4 * token_capacity_);
        if (cap_edges > edges_.capacity()) { edges_.reserve(cap_edges); }
    }

    void ingest(std::span<const Token> tokens) {
        reserve_additional(tokens.size() + 2);
        boundary();
        for (const auto t : tokens) {
            append(t);
        }
        boundary();
    }

    // O(1) online lookup from the generation tip (last_): walks suffix links to find the longest
    // suffix that appeared earlier in the live window with a valid non-boundary continuation.
    [[nodiscard]] std::optional<LookupResult> lookup() const noexcept {
        if (nodes_.empty() || last_ == 0) { return std::nullopt; }

        int32_t state      = last_;
        int32_t best_state = 0;
        int32_t best_pos   = -1;
        const auto n_tokens = static_cast<int32_t>(tokens_.size());
        const auto live     = static_cast<int32_t>(live_begin());

        while (state != 0) {
            const int32_t cand = best_lookup_candidate(state, nodes_[state].len, live, n_tokens);
            if (cand >= 0) {
                best_state = state;
                best_pos   = cand;
                break;
            }
            if (nodes_[state].link == -1) { break; }
            state = nodes_[state].link;
        }

        if (best_state == 0 || best_pos < 0) { return std::nullopt; }
        return LookupResult{
            .pos = best_pos + 1,
            .len = nodes_[best_state].len
        };
    }

    void advance_cursor(Cursor& c, Token t) const noexcept {
        if (c.generation != generation_) {
            c.state      = 0;
            c.matched    = 0;
            c.generation = generation_;
        }
        if (is_boundary(t)) {
            c.state   = 0;
            c.matched = 0;
            return;
        }
        int32_t state   = normalize_state(c.state, c.matched);
        int32_t matched = state == 0 ? 0 : c.matched;
        while (state != -1 && get_transition(state, t) == -1) {
            state   = nodes_[state].link;
            matched = state != -1 ? nodes_[state].len : 0;
        }
        if (state == -1) {
            c.state   = 0;
            c.matched = 0;
        } else {
            c.state   = get_transition(state, t);
            c.matched = matched + 1;
        }
    }

    [[nodiscard]] Cursor cursor_from(std::span<const Token> history) const noexcept {
        Cursor c{0, 0, generation_};
        for (const auto t : history) { advance_cursor(c, t); }
        return c;
    }

    [[nodiscard]] Match propose_from_cursor(const Cursor& c, std::uint32_t maximum,
                                            std::uint32_t minimum_match = 12) const {
        if (maximum == 0 || c.generation != generation_ || c.state == 0) { return {}; }
        const int32_t state = normalize_state(c.state, c.matched);
        return propose_from_state(state, c.matched, maximum, minimum_match);
    }

    [[nodiscard]] Match propose(std::span<const Token> history, std::uint32_t maximum,
                                std::uint32_t minimum_match = 12) const {
        Match match;
        if (maximum == 0 || history.empty()) { return match; }

        // Fast path: if history matches the active tail ending at last_ (with no boundary tokens),
        // last_ is already the SAM state for history, avoiding root-to-tip transition lookup.
        int32_t state   = 0;
        int32_t matched = 0;
        bool tail_match = (last_ != 0 && history.size() <= tokens_.size());
        if (tail_match) {
            const auto tail_offset = tokens_.size() - history.size();
            for (std::size_t i = 0; i < history.size(); ++i) {
                const auto t = history[i];
                if (t != tokens_[tail_offset + i] || is_boundary(t)) {
                    tail_match = false;
                    break;
                }
            }
        }
        if (tail_match) {
            state = last_;
            const auto hlen = static_cast<int32_t>(history.size());
            while (state != 0 && nodes_[state].link != -1 && nodes_[nodes_[state].link].len >= hlen) {
                state = nodes_[state].link;
            }
            matched = std::min<int32_t>(nodes_[state].len, hlen);
        } else {
            for (const auto t : history) {
                if (is_boundary(t)) {
                    state   = 0;
                    matched = 0;
                    continue;
                }
                while (state != -1 && get_transition(state, t) == -1) {
                    state   = nodes_[state].link;
                    matched = state != -1 ? nodes_[state].len : 0;
                }
                if (state == -1) {
                    state   = 0;
                    matched = 0;
                } else {
                    state = get_transition(state, t);
                    matched++;
                }
            }
        }
        return propose_from_state(state, matched, maximum, minimum_match);
    }

    [[nodiscard]] Match propose_from_state(int32_t state, int32_t matched, std::uint32_t maximum,
                                           std::uint32_t minimum_match) const {
        Match match;
        if (maximum == 0 || state <= 0 || matched < static_cast<int32_t>(minimum_match)) {
            return match;
        }

        const auto n_tokens = static_cast<int32_t>(tokens_.size());
        const auto live     = static_cast<int32_t>(live_begin());
        int32_t best_pos    = -1;
        int32_t alt_pos     = -1;
        uint32_t alt_freq   = 0;

        while (state != 0 && matched >= static_cast<int32_t>(minimum_match)) {
            best_pos = best_lookup_candidate(state, matched, live, n_tokens, &alt_pos, nullptr, &alt_freq);
            if (best_pos >= 0) { break; }
            state = nodes_[state].link;
            if (state > 0) { matched = std::min(matched, nodes_[state].len); }
        }

        if (state <= 0 || matched < static_cast<int32_t>(minimum_match) || best_pos < 0) {
            return match;
        }

        const int32_t match_end = best_pos + 1;
        uint32_t avail          = 0;
        while (avail < maximum && match_end + static_cast<int32_t>(avail) < n_tokens &&
               !is_boundary(tokens_[match_end + static_cast<int32_t>(avail)])) {
            avail++;
        }

        if (avail == 0) { return match; }

        match.matched     = static_cast<uint32_t>(matched);
        match.occurrences = nodes_[state].occ_count;
        match.distance    = static_cast<uint32_t>(n_tokens - 1 - best_pos);
        if (alt_pos >= 0) {
            match.alt_token       = tokens_[alt_pos + 1];
            match.alt_occurrences = alt_freq;
        }
        match.tokens.resize(avail);
        for (uint32_t i = 0; i < avail; ++i) {
            match.tokens[i] = tokens_[match_end + i];
        }
        return match;
    }

    [[nodiscard]] Match propose_for_round(std::span<const Token> history, std::uint32_t maximum,
                                          std::uint32_t neural_drafts,
                                          std::uint32_t minimum_match = 12) const {
        if (maximum == 0) { return {}; }
        if (maximum > std::numeric_limits<std::uint32_t>::max() - 2U) {
            throw std::invalid_argument("ngram proposal lookahead overflows");
        }
        return finish_round(propose(history, maximum + 2U, minimum_match), maximum, neural_drafts);
    }

    [[nodiscard]] static Match finish_round(Match match, std::uint32_t maximum,
                                            std::uint32_t neural_drafts) {
        if (maximum == 0) { return {}; }
        if (maximum > std::numeric_limits<std::uint32_t>::max() - 2U) {
            throw std::invalid_argument("ngram proposal lookahead overflows");
        }
        const auto available = static_cast<std::uint32_t>(match.tokens.size());
        const auto drafts    = std::min(maximum, available > 2U ? available - 2U : 0U);
        if (available < maximum + 2U && drafts <= neural_drafts) { return {}; }
        match.tokens.resize(drafts);
        return match;
    }

    [[nodiscard]] static bool maximal(const Match& match, std::size_t history_size,
                                      std::uint32_t maximum) {
        return match.matched == history_size && match.tokens.size() == maximum;
    }

private:
    static constexpr int32_t kMaxPropagateHops = 256;

    struct Edge {
        Token key;
        int32_t to;
        int32_t next;
    };

    struct Node {
        int32_t len            = 0;
        int32_t link           = -1;
        int32_t last_pos       = -1;
        int32_t prev_pos       = -1;
        int32_t valid_pos      = -1;
        int32_t prev_valid_pos = -1;
        int32_t top_pos        = -1;
        int32_t alt_pos        = -1;
        uint32_t occ_count     = 0;
        Token first_key        = -1;
        int32_t first_to       = -1;
        int32_t next_edge      = -1;
    };

    struct EdgeSlot {
        int32_t u   = -1;
        Token key   = 0;
        int32_t to  = -1;
    };

    std::size_t token_capacity_ = 0;
    bool compact_enabled_       = true;
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::vector<EdgeSlot> edge_table_;
    std::size_t edge_table_count_ = 0;
    std::vector<Token> tokens_;
    std::vector<Token> boundaries_;
    int32_t last_            = 0;
    std::uint64_t generation_ = 1;

    [[nodiscard]] int32_t normalize_state(int32_t state, int32_t matched) const noexcept {
        if (state <= 0 || matched <= 0) { return 0; }
        while (state > 0 && nodes_[state].link >= 0 && nodes_[nodes_[state].link].len >= matched) {
            state = nodes_[state].link;
        }
        return state;
    }

    void append_non_boundary(Token token) {
        const int32_t prev_last = last_;
        const int32_t prev_end  = !tokens_.empty() ? static_cast<int32_t>(tokens_.size()) - 1 : -1;
        // The non-boundary token being appended confirms that the previous tip ended a valid
        // continuation for `last_` and its suffix ancestors.
        if (prev_last != 0 && prev_end >= 0) {
            int32_t u = prev_last;
            for (int32_t hops = 0; u > 0 && hops < kMaxPropagateHops; ++hops) {
                if (nodes_[u].valid_pos != prev_end) {
                    nodes_[u].prev_valid_pos = nodes_[u].valid_pos;
                    nodes_[u].valid_pos      = prev_end;
                }
                u = nodes_[u].link;
            }
        }

        tokens_.push_back(token);
        const auto cur_pos = static_cast<int32_t>(tokens_.size()) - 1;
        const auto cur     = static_cast<int32_t>(nodes_.size());
        nodes_.push_back(Node{
            .len            = nodes_[last_].len + 1,
            .link           = 0,
            .last_pos       = cur_pos,
            .prev_pos       = -1,
            .valid_pos      = -1,
            .prev_valid_pos = -1,
            .occ_count      = 1,
            .first_key      = -1,
            .first_to       = -1,
            .next_edge      = -1
        });

        int32_t p = last_;
        while (p != -1 && get_transition(p, token) == -1) {
            set_transition(p, token, cur);
            p = nodes_[p].link;
        }

        if (p == -1) {
            nodes_[cur].link = 0;
        } else {
            const int32_t q = get_transition(p, token);
            if (nodes_[p].len + 1 == nodes_[q].len) {
                nodes_[cur].link = q;
            } else {
                const auto clone = static_cast<int32_t>(nodes_.size());
                nodes_.push_back(Node{
                    .len            = nodes_[p].len + 1,
                    .link           = nodes_[q].link,
                    .last_pos       = nodes_[q].last_pos,
                    .prev_pos       = nodes_[q].prev_pos,
                    .valid_pos      = nodes_[q].valid_pos,
                    .prev_valid_pos = nodes_[q].prev_valid_pos,
                    .top_pos        = nodes_[q].top_pos,
                    .alt_pos        = nodes_[q].alt_pos,
                    .occ_count      = nodes_[q].occ_count,
                    .first_key      = -1,
                    .first_to       = -1,
                    .next_edge      = -1
                });
                copy_transitions(clone, q);

                int32_t p_curr = p;
                while (p_curr != -1 && get_transition(p_curr, token) == q) {
                    set_transition(p_curr, token, clone);
                    p_curr = nodes_[p_curr].link;
                }
                nodes_[q].link   = clone;
                nodes_[cur].link = clone;
            }
        }

        // Propagate the new end position and occurrence count up the suffix-link chain so lookup
        // and propose see the most recent occurrence at every matching state.
        int32_t u = nodes_[cur].link;
        for (int32_t hops = 0; u > 0 && hops < kMaxPropagateHops; ++hops) {
            if (nodes_[u].last_pos != cur_pos) {
                nodes_[u].prev_pos = nodes_[u].last_pos;
                nodes_[u].last_pos = cur_pos;
            }
            if (nodes_[u].occ_count < std::numeric_limits<uint32_t>::max()) {
                ++nodes_[u].occ_count;
            }
            u = nodes_[u].link;
        }

        // Maintain top-2 highest-frequency distinct continuation slots (`top_pos`, `alt_pos`) on
        // each ancestor of the previous tip after transition occurrence counts have settled.
        if (prev_last != 0 && prev_end >= 0) {
            const int32_t live_floor = static_cast<int32_t>(live_begin());
            const auto freq_at = [&](int32_t node, int32_t pos) noexcept -> uint32_t {
                if (pos < live_floor || pos + 1 >= static_cast<int32_t>(tokens_.size()) ||
                    is_boundary(tokens_[pos + 1])) {
                    return 0U;
                }
                const int32_t nx = get_transition(node, tokens_[pos + 1]);
                return nx > 0 ? nodes_[nx].occ_count : 1U;
            };
            int32_t pu = prev_last;
            for (int32_t hops = 0; pu > 0 && hops < kMaxPropagateHops; ++hops) {
                const uint32_t f_new = freq_at(pu, prev_end);
                Node& pnd            = nodes_[pu];
                const uint32_t f_top = freq_at(pu, pnd.top_pos);
                if (pnd.top_pos < 0 || f_top == 0U || tokens_[pnd.top_pos + 1] == token) {
                    pnd.top_pos = prev_end;
                } else if (f_new > f_top) {
                    pnd.alt_pos = pnd.top_pos;
                    pnd.top_pos = prev_end;
                } else {
                    const uint32_t f_alt = freq_at(pu, pnd.alt_pos);
                    if (pnd.alt_pos < 0 || f_alt == 0U || tokens_[pnd.alt_pos + 1] == token || f_new >= f_alt) {
                        pnd.alt_pos = prev_end;
                    }
                }
                pu = nodes_[pu].link;
            }
        }

        last_ = cur;
    }

    void compact_if_needed() {
        if (!compact_enabled_) { return; }
        const std::size_t trigger = std::max<std::size_t>(2U * token_capacity_, 1024U);
        if (tokens_.size() < trigger) { return; }
        const std::size_t keep = std::min(token_capacity_, tokens_.size());
        std::vector<Token> tail(tokens_.end() - static_cast<std::ptrdiff_t>(keep), tokens_.end());
        clear();
        reserve_additional(tail.size());
        for (const Token t : tail) {
            if (is_boundary(t)) {
                tokens_.push_back(-1);
                last_ = 0;
            } else {
                append_non_boundary(t);
            }
        }
    }

    [[nodiscard]] int32_t best_lookup_candidate(int32_t state, int32_t matched, int32_t live,
                                                int32_t n_tokens, int32_t* alt_pos = nullptr,
                                                uint32_t* best_freq_out = nullptr,
                                                uint32_t* alt_freq_out = nullptr) const noexcept {
        const Node& nd     = nodes_[state];
        int32_t best       = -1;
        uint32_t best_freq = 0;
        const int32_t cands[6]{nd.top_pos, nd.alt_pos, nd.valid_pos, nd.prev_valid_pos, nd.last_pos, nd.prev_pos};
        for (const int32_t p : cands) {
            if (p >= 0 && p + 1 < n_tokens && p - matched + 1 >= live && !is_boundary(tokens_[p + 1])) {
                const int32_t next_state = get_transition(state, tokens_[p + 1]);
                const uint32_t freq      = next_state > 0 ? nodes_[next_state].occ_count : 1U;
                if (freq > best_freq || (freq == best_freq && p > best)) {
                    best_freq = freq;
                    best      = p;
                }
            }
        }
        if (best_freq_out != nullptr) { *best_freq_out = best_freq; }
        if (alt_pos != nullptr) {
            int32_t alt       = -1;
            uint32_t alt_freq = 0;
            if (best >= 0) {
                const Token best_tok = tokens_[best + 1];
                for (const int32_t p : cands) {
                    if (p >= 0 && p + 1 < n_tokens && p - matched + 1 >= live &&
                        !is_boundary(tokens_[p + 1]) && tokens_[p + 1] != best_tok) {
                        const int32_t next_state = get_transition(state, tokens_[p + 1]);
                        const uint32_t freq      = next_state > 0 ? nodes_[next_state].occ_count : 1U;
                        if (freq > alt_freq || (freq == alt_freq && p > alt)) {
                            alt_freq = freq;
                            alt      = p;
                        }
                    }
                }
            }
            *alt_pos = alt;
            if (alt_freq_out != nullptr) { *alt_freq_out = alt_freq; }
        }
        return best;
    }

    [[nodiscard]] static std::size_t hash_edge(int32_t u, Token key) noexcept {
        std::uint64_t x = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32) |
                          static_cast<std::uint32_t>(key);
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return static_cast<std::size_t>(x);
    }

    void rehash_edge_table(std::size_t new_cap) {
        std::vector<EdgeSlot> next(new_cap);
        const std::size_t mask = new_cap - 1;
        for (const auto& s : edge_table_) {
            if (s.u < 0) { continue; }
            std::size_t idx = hash_edge(s.u, s.key) & mask;
            while (next[idx].u >= 0) { idx = (idx + 1) & mask; }
            next[idx] = s;
        }
        edge_table_ = std::move(next);
    }

    void put_edge_table(int32_t u, Token token, int32_t to) {
        if ((edge_table_count_ + 1) * 2 > edge_table_.size()) {
            rehash_edge_table(edge_table_.empty() ? 1024 : edge_table_.size() * 2);
        }
        const std::size_t mask = edge_table_.size() - 1;
        std::size_t idx = hash_edge(u, token) & mask;
        while (edge_table_[idx].u >= 0) {
            if (edge_table_[idx].u == u && edge_table_[idx].key == token) {
                edge_table_[idx].to = to;
                return;
            }
            idx = (idx + 1) & mask;
        }
        edge_table_[idx] = EdgeSlot{u, token, to};
        ++edge_table_count_;
    }

    [[nodiscard]] static bool power_of_two(std::size_t value) noexcept {
        return value != 0 && (value & (value - 1)) == 0;
    }

    [[nodiscard]] std::uint64_t live_begin() const noexcept {
        return tokens_.size() > token_capacity_ ? tokens_.size() - token_capacity_ : 0;
    }

    [[nodiscard]] int32_t get_transition(int32_t u, Token token) const noexcept {
        if (nodes_[u].first_key == token) { return nodes_[u].first_to; }
        if (nodes_[u].next_edge == -1 || edge_table_.empty()) { return -1; }
        const std::size_t mask = edge_table_.size() - 1;
        std::size_t idx = hash_edge(u, token) & mask;
        while (edge_table_[idx].u >= 0) {
            if (edge_table_[idx].u == u && edge_table_[idx].key == token) {
                return edge_table_[idx].to;
            }
            idx = (idx + 1) & mask;
        }
        return -1;
    }

    void set_transition(int32_t u, Token token, int32_t to) {
        if (nodes_[u].first_key == token) {
            nodes_[u].first_to = to;
            return;
        }
        if (nodes_[u].first_key == -1) {
            nodes_[u].first_key = token;
            nodes_[u].first_to  = to;
            return;
        }
        if (nodes_[u].next_edge != -1 && !edge_table_.empty()) {
            const std::size_t mask = edge_table_.size() - 1;
            std::size_t idx = hash_edge(u, token) & mask;
            while (edge_table_[idx].u >= 0) {
                if (edge_table_[idx].u == u && edge_table_[idx].key == token) {
                    edge_table_[idx].to = to;
                    return;
                }
                idx = (idx + 1) & mask;
            }
        }
        const auto new_edge = static_cast<int32_t>(edges_.size());
        edges_.push_back(Edge{token, to, nodes_[u].next_edge});
        nodes_[u].next_edge = new_edge;
        put_edge_table(u, token, to);
    }

    void copy_transitions(int32_t dst, int32_t src) {
        nodes_[dst].first_key = nodes_[src].first_key;
        nodes_[dst].first_to  = nodes_[src].first_to;
        nodes_[dst].next_edge = -1;
        for (int32_t e = nodes_[src].next_edge; e != -1; e = edges_[e].next) {
            const auto new_edge = static_cast<int32_t>(edges_.size());
            edges_.push_back(Edge{edges_[e].key, edges_[e].to, nodes_[dst].next_edge});
            nodes_[dst].next_edge = new_edge;
            put_edge_table(dst, edges_[e].key, edges_[e].to);
        }
    }
};

// Non-blocking dual-generation SAM: swaps active and standby banks in O(1) when the active bank
// reaches capacity, eliminating stop-the-world 262K-token rebuild spikes during long runs while
// keeping the prior bank warm for immediate queries.
class DualGenSuffixAutomaton {
public:
    using Token = SuffixAutomaton::Token;
    using Match = SuffixAutomaton::Match;

    struct Cursor {
        SuffixAutomaton::Cursor active{};
        SuffixAutomaton::Cursor standby{};
        std::uint64_t generation = 0;
    };

    explicit DualGenSuffixAutomaton(std::size_t total_capacity = 1U << 18)
        : bank_capacity_(bank_cap_for(total_capacity)),
          active_(bank_capacity_, /*compact_enabled=*/false),
          standby_(bank_capacity_, /*compact_enabled=*/false) {}

    [[nodiscard]] std::size_t capacity() const noexcept { return bank_capacity_ * 2; }
    [[nodiscard]] std::size_t bank_capacity() const noexcept { return bank_capacity_; }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] std::size_t active_size() const noexcept { return active_.size(); }
    [[nodiscard]] std::size_t standby_size() const noexcept { return standby_.size(); }

    void clear() {
        active_.clear();
        standby_.clear();
        ++generation_;
    }

    void set_boundaries(std::vector<Token> tokens) {
        active_.set_boundaries(tokens);
        standby_.set_boundaries(std::move(tokens));
    }

    void ingest(std::span<const Token> tokens) {
        if (tokens.empty()) { return; }
        const std::size_t max_single = bank_capacity_ > 2 ? bank_capacity_ - 2 : tokens.size();
        if (tokens.size() > max_single) {
            tokens = tokens.last(max_single);
        }
        const std::size_t need = tokens.size() + 2;
        if (active_.size() > 0 && active_.size() + need > bank_capacity_) {
            std::swap(standby_, active_);
            active_.clear();
            ++generation_;
        }
        active_.ingest(tokens);
    }

    void advance_cursor(Cursor& c, Token token) const noexcept {
        if (c.generation != generation_) { return; }
        active_.advance_cursor(c.active, token);
        standby_.advance_cursor(c.standby, token);
    }

    [[nodiscard]] Cursor cursor_from(std::span<const Token> history) const noexcept {
        return Cursor{
            .active     = active_.cursor_from(history),
            .standby    = standby_.cursor_from(history),
            .generation = generation_,
        };
    }

    [[nodiscard]] Match propose(std::span<const Token> history, std::uint32_t maximum,
                                std::uint32_t minimum_match = 12) const {
        return pick_better(active_.propose(history, maximum, minimum_match),
                           standby_.propose(history, maximum, minimum_match));
    }

    [[nodiscard]] Match propose_from_cursor(const Cursor& c, std::uint32_t maximum,
                                            std::uint32_t minimum_match = 12) const {
        if (c.generation != generation_) { return {}; }
        return pick_better(active_.propose_from_cursor(c.active, maximum, minimum_match),
                           standby_.propose_from_cursor(c.standby, maximum, minimum_match));
    }

private:
    std::size_t bank_capacity_ = 0;
    SuffixAutomaton active_;
    SuffixAutomaton standby_;
    std::uint64_t generation_ = 1;

    [[nodiscard]] static std::size_t bank_cap_for(std::size_t total_capacity) {
        if (total_capacity < 128 || (total_capacity & (total_capacity - 1)) != 0) {
            throw std::invalid_argument("invalid dual-gen suffix automaton capacity");
        }
        return std::max<std::size_t>(total_capacity / 2, 64);
    }

    [[nodiscard]] static Match pick_better(Match a, Match b) noexcept {
        if (a.tokens.empty()) { return b; }
        if (!b.tokens.empty() &&
            (b.matched > a.matched ||
             (b.matched == a.matched &&
              (b.occurrences > a.occurrences ||
               (b.occurrences == a.occurrences && b.tokens.size() > a.tokens.size()))))) {
            return b;
        }
        return a;
    }
};

} // namespace infernix::models::qwen3_5::detail