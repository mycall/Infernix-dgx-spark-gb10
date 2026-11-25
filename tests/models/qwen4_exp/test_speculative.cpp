// The MTP draft round's host policy (models/qwen4_exp/program/mtp_gating.h): the cumulative cut and
// the executed-draft-step count it reports (what a gated chain paid for), and the copy-versus-chain
// choice that lets a round the cut truncated still verify the n-gram proposal it declined before the
// drafter ran. Every expectation is checked against a direct evaluation of the same rule, and the cut
// against a simulation of the device chain (core/decode_graph.h's launch_graph_gate_product), which
// must run exactly the steps the host believes were paid for. The cut step is never verified, even
// though a gated chain ran it — retaining it was measured net-negative on GB10 (mtp_gating.h), so
// the kept count stopping at the cut is itself a regression guard.

#include "models/qwen4_exp/program/mtp_gating.h"
#include "models/qwen3_5/program/ngram_proposer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using infernix::GraphGateBins;
using infernix::graph_gate_product;
using infernix::graph_gate_rate;
using infernix::models::qwen4_exp::copy_beats_chain;
using infernix::models::qwen4_exp::expected_chain_tokens;
using infernix::models::qwen4_exp::expected_copy_tokens;
using infernix::models::qwen4_exp::kMtpCumulativeBins;
using infernix::models::qwen4_exp::kMtpCumulativeBase;
using infernix::models::qwen4_exp::kMtpCumulativeFloor;
using infernix::models::qwen4_exp::kMtpSampledCumulativeBins;
using infernix::models::qwen4_exp::kMtpSampledCumulativeBase;
using infernix::models::qwen4_exp::kMtpSampledCumulativeFloor;
using infernix::models::qwen4_exp::MtpCumulative;
using infernix::models::qwen4_exp::MtpDraftCut;
using infernix::models::qwen4_exp::mtp_cumulative_cut;
using infernix::models::qwen4_exp::mtp_min_logprob_cut;

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error(what); }
}

void require_cut(const MtpDraftCut& cut, std::int32_t kept, std::int32_t executed, const std::string& what) {
    require(cut.kept == kept && cut.executed == executed,
            what + " (kept " + std::to_string(cut.kept) + ", executed " + std::to_string(cut.executed) +
                "; expected " + std::to_string(kept) + ", " + std::to_string(executed) + ")");
}

// One log-probability per bin of the shipped table; the rates the tests quote are those bins' rates.
constexpr float kRateFloorLogprob = -3.00F;  //  0.12
constexpr float kRateMidLogprob   = -0.30F;  //  0.51
constexpr float kRateGoodLogprob  = -0.10F;  //  0.66
constexpr float kRateTopLogprob   = -0.005F; //  0.96

// The chain the device runs: body j runs, and body j + 1 only while steps 0..j's product clears
// `threshold` (launch_graph_gate_product clears the conditional handle below it). Step i's value sits
// at values[i * stride], the draft round's row pitch.
std::int32_t simulated_device_steps(const float* values, std::int32_t stride, std::int32_t steps, float threshold,
                                    const GraphGateBins& bins) {
    std::int32_t ran = 0;
    for (std::int32_t j = 0; j < steps; ++j) {
        ++ran;
        if (graph_gate_product(values, stride, j, bins) < threshold) { break; }
    }
    return ran;
}

// The first j whose chain product falls below `threshold`, or `steps` when none does.
std::int32_t brute_cut(const float* values, std::int32_t stride, std::int32_t steps, float threshold,
                       const GraphGateBins& bins) {
    for (std::int32_t j = 0; j < steps; ++j) {
        if (graph_gate_product(values, stride, j, bins) < threshold) { return j; }
    }
    return steps;
}

// Expected tokens of verifying `count` positions that each survive with probability `p`.
double brute_copy_tokens(std::int32_t count, double p) {
    double tokens = 1.0, reach = 1.0;
    for (std::int32_t j = 0; j < count; ++j) {
        reach *= p;
        tokens += reach;
    }
    return tokens;
}

struct Rng {
    std::uint32_t state = 12345U;
    float next_logprob() {
        state = state * 1664525U + 1013904223U;
        return -6.0F + 6.5F * (static_cast<float>(state >> 8) / static_cast<float>(1U << 24));
    }
    double next_unit() {
        state = state * 1664525U + 1013904223U;
        return static_cast<double>(state >> 8) / static_cast<double>(1U << 24);
    }
};

// The shipped table, read from the header: the rates the tests below quote really are its rates.
void check_table() {
    require(graph_gate_rate(kRateFloorLogprob, kMtpCumulativeBins) == 0.12F, "the cheapest bin is 0.12");
    require(graph_gate_rate(kRateMidLogprob, kMtpCumulativeBins) == 0.51F, "the middle bin is 0.51");
    require(graph_gate_rate(kRateGoodLogprob, kMtpCumulativeBins) == 0.66F, "the good bin is 0.66");
    require(graph_gate_rate(kRateTopLogprob, kMtpCumulativeBins) == 0.96F, "the top bin is 0.96");
    require(kMtpCumulativeBase == 0.40F, "the continuation threshold starts at 0.40");
}

// Where the continuation cut lands, and which steps a gated chain paid for. The cut step is dropped
// even though it ran: its acceptance is conditioned on the chain being cut there and was measured
// net-negative to verify (mtp_gating.h) — kept stopping at the cut is the regression guard.
void check_continuation_cut() {
    // 0.66, then 0.66 * 0.66 = 0.4356, then 0.2875: the gate stops the chain on step 2, which ran but
    // is not verified, so the round verifies steps 0..1.
    const std::vector<float> chain{kRateGoodLogprob, kRateGoodLogprob, kRateGoodLogprob, kRateGoodLogprob};
    require_cut(mtp_cumulative_cut(chain.data(), 1, 4, kMtpCumulativeBase, kMtpCumulativeBins, true), 2, 3,
                "a gated chain verifies up to but not including its cut step");
    // The same chain on an ungated round (multi-row, single-step or sampled): every drafted step ran,
    // so the executed count is the drafted count even though the host cuts at the same place.
    require_cut(mtp_cumulative_cut(chain.data(), 1, 4, kMtpCumulativeBase, kMtpCumulativeBins, false), 2, 4,
                "an ungated chain ran every step it drafted");
    // The request-adapted threshold (the 0.25 floor) moves the cut later: 0.66 to the 4th is 0.19, so
    // the chain is cut on step 3 and verifies three drafts.
    require_cut(mtp_cumulative_cut(chain.data(), 1, 4, 0.25F, kMtpCumulativeBins, true), 3, 4,
                "the adapted threshold cuts one step later");
    // A threshold above every product cuts at step 0, which always ran and is never verified.
    require_cut(mtp_cumulative_cut(chain.data(), 1, 4, 0.9F, kMtpCumulativeBins, true), 0, 1,
                "a chain cut on its first step verifies nothing");
    require_cut(mtp_cumulative_cut(chain.data(), 1, 4, 0.9F, kMtpCumulativeBins, false), 0, 4,
                "an ungated chain cut on its first step ran them all");
    // No cut at all: everything drafted is verified and ran.
    const std::vector<float> sure{kRateTopLogprob, kRateTopLogprob, kRateTopLogprob, kRateTopLogprob};
    require_cut(mtp_cumulative_cut(sure.data(), 1, 4, kMtpCumulativeBase, kMtpCumulativeBins, true), 4, 4,
                "an uncut chain keeps every draft");
    // A collapsed prefix cuts just as surely: 0.51, then 0.51 * 0.12 = 0.0612.
    const std::vector<float> collapsed{kRateMidLogprob, kRateFloorLogprob};
    require_cut(mtp_cumulative_cut(collapsed.data(), 1, 2, kMtpCumulativeBase, kMtpCumulativeBins, true), 1, 2,
                "a chain cut on a collapsed prefix keeps only the step before it");
    // An empty round drafts nothing.
    require_cut(mtp_cumulative_cut(chain.data(), 1, 0, kMtpCumulativeBase, kMtpCumulativeBins, true), 0, 0,
                "no steps, no drafts");
}

// The fixed --spec-min-p cut: an explicit floor on what a round may verify; the executed count still
// follows whether the device gated the chain.
void check_min_logprob_cut() {
    const std::vector<float> chain{-0.1F, -0.2F, -0.9F, -0.1F};
    require_cut(mtp_min_logprob_cut(chain.data(), 1, 4, -0.5F, true), 2, 3, "the fixed cut drops the floor draft");
    require_cut(mtp_min_logprob_cut(chain.data(), 1, 4, -0.5F, false), 2, 4, "an ungated fixed cut ran every step");
    require_cut(mtp_min_logprob_cut(chain.data(), 1, 4, -1.0F, true), 4, 4, "a floor below every draft cuts nothing");
    const std::vector<float> all_below{-0.6F, -0.6F};
    require_cut(mtp_min_logprob_cut(all_below.data(), 1, 2, -0.5F, true), 0, 1, "a floor above every draft cuts all");
}

// The rows of a multi-row round read their own drafts through the round's stride.
void check_strided_rows() {
    const std::vector<float> logprobs{// step-major: [row0 step0, row1 step0, row0 step1, ...]
                                      kRateGoodLogprob, kRateTopLogprob, kRateGoodLogprob, kRateTopLogprob,
                                      kRateGoodLogprob, kRateTopLogprob, kRateGoodLogprob, kRateTopLogprob};
    require_cut(mtp_cumulative_cut(logprobs.data(), 2, 4, kMtpCumulativeBase, kMtpCumulativeBins, false), 2, 4,
                "row 0 reads its own chain");
    require_cut(mtp_cumulative_cut(logprobs.data() + 1, 2, 4, kMtpCumulativeBase, kMtpCumulativeBins, false), 4, 4,
                "row 1 reads its own chain");
}

// The cut against a direct evaluation and against the device chain, over random chains and thresholds.
void check_against_device_chain() {
    Rng rng;
    for (int trial = 0; trial < 20000; ++trial) {
        const std::int32_t steps  = 1 + trial % 8;
        const std::int32_t rows   = 1 + trial % 3;
        std::vector<float> logprobs(static_cast<std::size_t>(steps) * rows);
        for (float& v : logprobs) { v = rng.next_logprob(); }
        const float threshold = 0.05F + 0.9F * static_cast<float>(rng.next_unit());
        const bool gated      = trial % 2 == 0;
        for (std::int32_t row = 0; row < rows; ++row) {
            const float* values       = logprobs.data() + row;
            const std::int32_t stride = rows;
            const auto cut            = mtp_cumulative_cut(values, stride, steps, threshold, kMtpCumulativeBins, gated);
            const std::string at      = "trial " + std::to_string(trial) + " row " + std::to_string(row);
            const std::int32_t first  = brute_cut(values, stride, steps, threshold, kMtpCumulativeBins);
            require(cut.kept >= 0 && cut.kept <= cut.executed && cut.executed <= steps, at + ": bounds");
            require(cut.kept == first, at + ": the cut drops its own step — nothing past the cut is verified");
            require(cut.executed == (gated ? std::min(first + 1, steps) : steps), at + ": the executed count");
            // Only a one-row round runs under the device gate (mtp_draft_gated); there the host's
            // executed count must be exactly the steps its conditional chain ran.
            if (gated && rows == 1) {
                require(cut.executed == simulated_device_steps(values, stride, steps, threshold, kMtpCumulativeBins),
                        at + ": the device chain ran that many steps");
            }
        }
    }
}

// A truncated round still verifies a copy: the chain side of the comparison is the kept prefix, so a
// row the cut left with nothing takes any copy whose own acceptance beats its column cost.
void check_copy_fallback() {
    const std::array<double, 8> good{0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95};
    const std::array<double, 8> poor{0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30};
    constexpr double kCopyCost = 0.12, kChainCost = 0.28;

    // A row the cut left with no draft: its chain is worth the one target token, so a copy of three at
    // 0.75 (2.73 tokens for 1.36 rounds) is verified instead of the round decaying to a plain one.
    require(copy_beats_chain(3, 0.75, kCopyCost, good, 0, kChainCost), "an empty chain loses to a good copy");
    require(copy_beats_chain(1, 0.75, kCopyCost, good, 0, kChainCost), "even a one-token copy beats no draft");
    // Break-even: one copy at exactly its column cost earns nothing over the plain round.
    require(!copy_beats_chain(1, kCopyCost, kCopyCost, good, 0, kChainCost), "a copy at its column cost is not taken");
    require(!copy_beats_chain(4, 0.05, kCopyCost, good, 0, kChainCost), "a copy nobody accepts is not taken");
    // A chain the cut left intact still wins against a poor copy, and a good copy still wins against a
    // chain whose own acceptance is poor.
    require(!copy_beats_chain(2, 0.4, kCopyCost, good, 4, kChainCost), "a whole good chain beats a poor copy");
    require(copy_beats_chain(4, 0.9, kCopyCost, poor, 4, kChainCost), "a good copy beats a poor chain");
    // The post-cut comparison is the pre-drafter one over the kept prefix: as the cut keeps more of a
    // good chain, the round stops preferring the same five-token copy at 0.7.
    for (std::int32_t drafts = 0; drafts <= 8; ++drafts) {
        const bool take = copy_beats_chain(5, 0.7, kCopyCost, good, drafts, kChainCost);
        require(take == (drafts <= 2), "a five-token copy wins only while the chain is short (drafts " +
                                           std::to_string(drafts) + ")");
    }
    // A copy longer than the round's whole chain is taken whenever it earns more per round time, which
    // is the same rule before the drafter ran and after its cut.
    require(copy_beats_chain(8, 0.9, kCopyCost, good, 2, kChainCost), "a long good copy beats a short chain");
    require(!copy_beats_chain(8, 0.2, kCopyCost, good, 2, kChainCost), "a long poor copy does not");

    // Acceptance-floored best_copy_length: below the floor, no copy drafts are admitted; above the
    // floor, length is capped so expected acceptance stays >= min_accept while high-confidence
    // copies still reach the full window.
    using infernix::models::qwen4_exp::best_copy_length;
    require(best_copy_length(31, 0.55, kCopyCost, 0.60) == 0, "below min_accept floor yields zero drafts");
    const std::int32_t mid = best_copy_length(31, 0.72, kCopyCost, 0.60);
    require(mid >= 1 && mid <= 3, "moderate prior probes short instead of drafting 10+ tokens");
    const std::int32_t high = best_copy_length(31, 0.985, kCopyCost, 0.60);
    require(high >= 24, "confirmed high-confidence copy still expands to a wide window");
    require(best_copy_length(31, 0.995, kCopyCost, 0.60) == 31,
            "near-exact copy reaches the full 31-token window");
}

void check_moe_round_cost_and_fit() {
    using infernix::models::qwen4_exp::RoundCostModel;
    using infernix::models::qwen4_exp::moe_expected_unique_experts;
    using infernix::models::qwen4_exp::moe_round_cost;

    require(moe_expected_unique_experts(512, 10, 0) == 0.0, "0 columns -> 0 unique experts");
    require(moe_expected_unique_experts(16, 16, 4) == 16.0, "top_k == experts saturates all experts");
    const double u1 = moe_expected_unique_experts(512, 10, 1);
    const double u2 = moe_expected_unique_experts(512, 10, 2);
    const double u8 = moe_expected_unique_experts(512, 10, 8);
    require(std::abs(u1 - 10.0) < 1e-9, "1 token activates top_k experts");
    require(std::abs(u2 - 512.0 * (1.0 - std::pow(502.0 / 512.0, 2.0))) < 1e-9, "2 tokens match coupon formula");
    require(u2 > u1 && u8 > u2 && (u2 - u1) < u1, "coupon collector is strictly increasing and concave");

    require(moe_round_cost(1, 0, 512, 10, 45.0, 0.05) == 1.0, "plain round relative cost is 1.0");
    require(moe_round_cost(8, 0, 512, 10, 45.0, 0.05) == 1.0, "batch plain round relative cost is 1.0");
    const double c_b1_d4 = moe_round_cost(1, 4, 512, 10, 45.0, 0.05);
    const double c_b4_d4 = moe_round_cost(4, 4, 512, 10, 45.0, 0.05);
    const double c_b8_d4 = moe_round_cost(8, 4, 512, 10, 45.0, 0.05);
    require(c_b8_d4 > c_b4_d4 && c_b4_d4 > c_b1_d4,
            "drafting at higher concurrency increases relative MoE verify expansion cost");
    for (std::int32_t batch : {1, 4, 8}) {
        for (std::int32_t k : {1, 2, 4}) {
            const double ub1 = moe_expected_unique_experts(512, 10, 1);
            const double u0  = moe_expected_unique_experts(512, 10, batch);
            const double uk  = moe_expected_unique_experts(512, 10, batch * (1 + k));
            const double expected = (60.0 + uk) / (60.0 + u0) + 0.06 * ((60.0 + ub1) / (60.0 + u0)) * k;
            require(std::abs(moe_round_cost(batch, k, 512, 10, 60.0, 0.06) - expected) < 1e-12,
                    "moe_round_cost matches batch-normalized coupon formula");
        }
    }

    RoundCostModel model;
    double dense_fit = 0.0, step_fit = 0.0;
    require(!model.fit(512, 10, 45.0, 0.05, dense_fit, step_fit), "fit requires kMinRounds");

    constexpr std::int32_t kExperts   = 512;
    constexpr std::int32_t kTopK      = 10;
    constexpr double kTrueDenseEquiv  = 64.0;
    constexpr double kTrueStepCost    = 0.042;
    constexpr double kBasePlainMs     = 0.011;
    for (int round = 0; round < 160; ++round) {
        std::int32_t verified = 0;
        std::int32_t exec     = 0;
        switch (round % 4) {
        case 0: verified = 0; exec = 0; break;            // plain
        case 1: verified = 4; exec = 4; break;            // complete MTP
        case 2: verified = 1 + (round % 3); exec = 3; break; // gated MTP
        case 3: verified = 8 + (round % 9); exec = 0; break; // N-gram / SAM
        }
        const double u_base = moe_expected_unique_experts(kExperts, kTopK, 1);
        const double u_var  = moe_expected_unique_experts(kExperts, kTopK, 1 + verified);
        const double dt =
            kBasePlainMs * ((kTrueDenseEquiv + u_var) / (kTrueDenseEquiv + u_base) + kTrueStepCost * exec);
        model.observe(verified, exec, dt, kExperts, kTopK, 45.0, 0.05);
        if (round == 80) {
            // Inject an OS preemption spike (should be rejected by 3x outlier guard).
            model.observe(verified, exec, dt * 10.0, kExperts, kTopK, 45.0, 0.05);
        }
    }
    require(model.fit(kExperts, kTopK, 45.0, 0.05, dense_fit, step_fit), "fit succeeds after mixed rounds");
    require(std::abs(dense_fit - kTrueDenseEquiv) < 4.0,
            "RoundCostModel recovers dense_equiv across mixed B=1 traces (got " + std::to_string(dense_fit) + ")");
    require(std::abs(step_fit - kTrueStepCost) < 0.005,
            "RoundCostModel recovers draft_step_cost across mixed B=1 traces (got " + std::to_string(step_fit) + ")");

    // Rank-1 trace (all plain B=1 rounds): step_fit and dense_fit must stay safely at priors.
    RoundCostModel plain_only;
    for (int r = 0; r < 24; ++r) { plain_only.observe(0, 0, kBasePlainMs, kExperts, kTopK, 45.0, 0.05); }
    require(plain_only.fit(kExperts, kTopK, 45.0, 0.05, dense_fit, step_fit), "rank-1 trace still fits safely");
    require(std::abs(dense_fit - 45.0) < 1e-6 && std::abs(step_fit - 0.05) < 1e-6,
            "rank-1 plain trace preserves prior dense_equiv and draft_step_cost");
}

void check_ngram_buckets_and_moe_copy_stitch() {
    using infernix::models::qwen4_exp::best_copy_length_moe;
    using infernix::models::qwen4_exp::best_stitch_length;
    using infernix::models::qwen4_exp::copy_beats_chain_moe;
    using infernix::models::qwen4_exp::moe_round_cost;
    using infernix::models::qwen4_exp::ngram_confidence_bucket;

    require(ngram_confidence_bucket(0, 5, 10, false) == 0, "bucket 0 for matched 0");
    require(ngram_confidence_bucket(3, 5, 10, false) == 0, "bucket 0 for matched 3");
    require(ngram_confidence_bucket(5, 2, 10, false) == 0, "bucket 0 for matched 5 low occ");
    require(ngram_confidence_bucket(7, 2, 10, false) == 0, "bucket 0 for matched 7 low occ");
    require(ngram_confidence_bucket(5, 3, 10, false) == 1, "bucket 1 for matched 5 occ 3");
    require(ngram_confidence_bucket(7, 5, 10, false) == 1, "bucket 1 for matched 7 occ 5");
    require(ngram_confidence_bucket(6, 6, 10, false) == 2, "bucket 2 for matched 6 occ 6");
    require(ngram_confidence_bucket(8, 1, 4096, false) == 1, "bucket 1 for 8..11 single stale");
    require(ngram_confidence_bucket(11, 1, 2049, false) == 1, "bucket 1 just outside recency window");
    require(ngram_confidence_bucket(8, 2, 10000, false) == 2, "bucket 2 for 8..11 repeated");
    require(ngram_confidence_bucket(10, 1, 2048, false) == 2, "bucket 2 for 8..11 recent");
    require(ngram_confidence_bucket(12, 1, 5000, false) == 3, "bucket 3 for 12..15 single stale");
    require(ngram_confidence_bucket(15, 3, 9999, false) == 4, "bucket 4 for 12..15 repeated");
    require(ngram_confidence_bucket(13, 1, 512, false) == 4, "bucket 4 for 12..15 recent");
    require(ngram_confidence_bucket(16, 1, 5000, false) == 5, "bucket 5 for 16+ single stale");
    require(ngram_confidence_bucket(31, 4, 9000, false) == 6, "bucket 6 for 16+ repeated");
    require(ngram_confidence_bucket(20, 1, 1024, false) == 6, "bucket 6 for 16+ recent");
    require(ngram_confidence_bucket(3, 1, 90000, true) == 7, "bucket 7 whenever MTP confirmed/stitched");

    constexpr std::int32_t E = 512, K = 10;
    constexpr double D = 45.0, S = 0.05;
    require(best_copy_length_moe(31, 0.55, 1, E, K, D, 0.60) == 0, "below min_accept gives 0 copy drafts");
    const std::int32_t b1_mid = best_copy_length_moe(31, 0.80, 1, E, K, D, 0.60);
    const std::int32_t b8_mid = best_copy_length_moe(31, 0.80, 8, E, K, D, 0.60);
    require(b1_mid >= b8_mid, "higher concurrency tightens moderate-confidence copy length");
    require(best_copy_length_moe(31, 0.995, 1, E, K, D, 0.60) == 31, "high confidence reaches full 31 at B=1");

    // Exhaustive oracle check for best_copy_length_moe and best_stitch_length.
    Rng rng;
    for (int trial = 0; trial < 200; ++trial) {
        const std::int32_t batch  = 1 + (trial % 8);
        const std::int32_t copies = 1 + (trial % 31);
        const double copy_p       = 0.50 + 0.50 * rng.next_unit();
        constexpr double kMinAcc  = 0.60;

        std::int32_t brute_k = 0;
        double best_sc       = 1.0;
        double reach = 1.0, acc = 0.0;
        for (std::int32_t k = 1; k <= copies; ++k) {
            reach *= copy_p;
            acc += reach;
            if (acc < kMinAcc * static_cast<double>(k)) { break; }
            const double sc = (1.0 + acc) / moe_round_cost(batch, k, E, K, D, 0.0);
            if (sc > best_sc) {
                best_sc = sc;
                brute_k = k;
            }
        }
        require(best_copy_length_moe(copies, copy_p, batch, E, K, D, kMinAcc) == brute_k,
                "best_copy_length_moe matches brute-force maximization");

        const std::int32_t prefix_len = 1 + (trial % 4);
        const std::int32_t width_b    = std::max(prefix_len, 1 + (trial % 8));
        const double prefix_reach     = 0.4 + 0.55 * rng.next_unit();
        const double prefix_toks      = 1.0 + static_cast<double>(prefix_len) * prefix_reach;
        std::int32_t brute_s          = 0;
        double best_stitch_sc         = prefix_toks / moe_round_cost(batch, width_b, E, K, D, 0.0);
        double s_reach = 1.0, s_acc = 0.0;
        for (std::int32_t s = 1; s <= copies; ++s) {
            s_reach *= copy_p;
            s_acc += s_reach;
            if (s_acc < kMinAcc * static_cast<double>(s)) { break; }
            const double sc =
                (prefix_toks + prefix_reach * s_acc) /
                moe_round_cost(batch, std::max(width_b, prefix_len + s), E, K, D, 0.0);
            if (sc > best_stitch_sc) {
                best_stitch_sc = sc;
                brute_s        = s;
            }
        }
        require(best_stitch_length(prefix_len, prefix_toks, prefix_reach, copies, copy_p, batch, width_b, E, K, D,
                                   kMinAcc) == brute_s,
                "best_stitch_length matches brute-force maximization");
    }

    const std::array<double, 4> strong_mtp{0.92, 0.90, 0.88, 0.86};
    require(copy_beats_chain_moe(20, 0.99, strong_mtp, 3, 8, E, K, D, S),
            "near-exact 20-token copy beats 3-draft MTP chain even at B=8");
    require(!copy_beats_chain_moe(2, 0.72, strong_mtp, 3, 8, E, K, D, S),
            "short moderate copy loses to strong 3-draft MTP chain at B=8");
    // Free-width stitching: if batch width 8 is already paid by another row, stitching onto prefix_len=3
    // takes at least up to 5 free copy tokens when acceptance clears min_accept.
    const std::int32_t free_stitch = best_stitch_length(3, 3.2, 0.85, 8, 0.90, 4, 8, E, K, D, 0.60);
    require(free_stitch >= 5, "stitching exploits free already-paid batch width before capping");
}

// The two expected-token helpers against a direct evaluation.
void check_expected_tokens() {
    Rng rng;
    for (int trial = 0; trial < 2000; ++trial) {
        const std::int32_t count = trial % 9;
        const double p           = rng.next_unit();
        require(std::abs(expected_copy_tokens(count, p) - brute_copy_tokens(count, p)) < 1e-12, "copy tokens");
        std::array<double, 8> accept{};
        for (double& a : accept) { a = rng.next_unit(); }
        double expect = 1.0, reach = 1.0;
        for (std::int32_t j = 0; j < count; ++j) {
            reach *= accept[static_cast<std::size_t>(j)];
            expect += reach;
        }
        require(std::abs(expected_chain_tokens(std::span<const double>(accept), count) - expect) < 1e-12, "chain tokens");
        // A chain longer than the learned positions counts only the positions it has.
        require(std::abs(expected_chain_tokens(std::span<const double>(accept), 12) -
                         expected_chain_tokens(std::span<const double>(accept.data(), 8), 8)) < 1e-12,
                "a chain counts only the acceptances it has");
    }
}

void check_sampled_cumulative_table() {
    for (std::size_t i = 0; i < std::size(kMtpSampledCumulativeBins.edges); ++i) {
        if (i > 0) {
            require(kMtpSampledCumulativeBins.edges[i - 1] < kMtpSampledCumulativeBins.edges[i],
                    "sampled cumulative bin edges must strictly increase");
        }
        require(kMtpSampledCumulativeBins.rates[i] < kMtpSampledCumulativeBins.rates[i + 1],
                "sampled cumulative bin rates must strictly increase");
    }
    for (const float r : kMtpSampledCumulativeBins.rates) {
        require(r > 0.0F && r <= 1.0F, "sampled cumulative bin rates must lie in (0, 1]");
    }
    require(kMtpSampledCumulativeFloor < kMtpSampledCumulativeBase,
            "sampled cumulative floor must sit below base");
    MtpCumulative cum;
    cum.reset(kMtpSampledCumulativeBase);
    require(cum.threshold == kMtpSampledCumulativeBase, "reset sets sampled base threshold");
    // Feed perfect acceptance across moderate-confidence chains -> threshold lowers smoothly toward floor.
    const std::array<float, 4> mod_lp{-0.8F, -0.8F, -0.8F, -0.8F};
    for (int round = 0; round < 64; ++round) {
        cum.observe(mod_lp.data(), 4, 4, kMtpSampledCumulativeBins, kMtpSampledCumulativeBase,
                    kMtpSampledCumulativeFloor);
    }
    require(cum.threshold >= kMtpSampledCumulativeFloor && cum.threshold < kMtpSampledCumulativeBase,
            "high observed/expected acceptance lowers sampled threshold toward floor");
    // Check host/device cut simulation agreement over kMtpSampledCumulativeBins.
    Rng rng;
    for (int trial = 0; trial < 5000; ++trial) {
        const std::int32_t steps = 1 + trial % 8;
        std::vector<float> lps(static_cast<std::size_t>(steps));
        for (float& v : lps) { v = rng.next_logprob(); }
        const float threshold = kMtpSampledCumulativeFloor +
                                0.5F * static_cast<float>(rng.next_unit());
        const bool gated = (trial % 2) == 0;
        const auto cut   = mtp_cumulative_cut(lps.data(), 1, steps, threshold, kMtpSampledCumulativeBins, gated);
        require(cut.kept == brute_cut(lps.data(), 1, steps, threshold, kMtpSampledCumulativeBins),
                "sampled cumulative kept matches brute cut");
        if (gated) {
            require(cut.executed == simulated_device_steps(lps.data(), 1, steps, threshold, kMtpSampledCumulativeBins),
                    "sampled cumulative executed matches device gate simulation");
        }
    }
}

void check_root_fork_trees() {
    using infernix::ops::SpeculativeTreeRow;
    using infernix::models::qwen4_exp::bucket_verify_width;
    using infernix::models::qwen4_exp::build_root_fork_tree_row;
    using infernix::models::qwen4_exp::choose_root_fork_side;
    using infernix::models::qwen4_exp::floor_verify_width;
    using infernix::ops::kSpeculativeTreeMaxNodes;
    using infernix::ops::kSpeculativeTreeMaxPathLength;

    // 16 candidates with p[0]=0.55 < 0.90 and residual mass over candidates[1..].
    std::array<std::int32_t, 16> cands{};
    std::array<float, 16> q{};
    for (std::size_t i = 0; i < 16; ++i) {
        cands[i] = static_cast<std::int32_t>(1000 + i);
        q[i]     = i == 0 ? 0.55F : 0.03F;
    }

    // pos % 4 == 2 must suppress root-fork (QSA pooled-block ambiguity).
    const auto skip_mod = choose_root_fork_side(1000, cands, q, /*position=*/102, /*ratio=*/4, /*seed=*/42);
    require(!skip_mod.fork, "root-fork gates out pos % r == r - 2");

    // High confidence top-1 (>= 0.90) must suppress root-fork.
    auto high_q = q;
    high_q[0]   = 0.92F;
    for (std::size_t i = 1; i < 16; ++i) { high_q[i] = 0.005F; }
    const auto skip_conf = choose_root_fork_side(1000, cands, high_q, /*position=*/100, /*ratio=*/4, /*seed=*/42);
    require(!skip_conf.fork, "root-fork gates out high step-0 top-1 mass >= 0.90");

    // Deterministic without-replacement side draw != main_token on eligible positions.
    for (const std::int32_t pos : {100, 101, 103}) {
        const auto d1 = choose_root_fork_side(1000, cands, q, pos, 4, 42);
        const auto d2 = choose_root_fork_side(1000, cands, q, pos, 4, 42);
        require(d1.fork && d2.fork, "eligible position activates root-fork");
        require(d1.side_token == d2.side_token, "root-fork draw is deterministic in (seed, position)");
        require(d1.side_token != 1000 && d1.side_token >= 1001 && d1.side_token <= 1015,
                "root-fork draws without replacement from remaining candidates");
    }

    // Check SpeculativeTreeRow geometry for chain and root-fork rows.
    SpeculativeTreeRow fork_row{};
    build_root_fork_tree_row(fork_row, /*main_depth=*/3, /*has_side=*/true);
    require(fork_row.tree == 1 && fork_row.transport == 0 && fork_row.sampled == 0,
            "root-fork sets tree=1 transport=0 sampled=0");
    require(fork_row.nodes == 5 && fork_row.main_depth == 3 && fork_row.paths == 2,
            "root-fork sets nodes/main_depth/paths");
    require(fork_row.parent[0] == -1 && fork_row.parent[1] == 0 && fork_row.parent[2] == 1 &&
                fork_row.parent[3] == 2 && fork_row.parent[4] == 0,
            "root-fork parent links are exact");
    require(fork_row.depth[0] == 0 && fork_row.depth[1] == 1 && fork_row.depth[2] == 2 &&
                fork_row.depth[3] == 3 && fork_row.depth[4] == 1,
            "root-fork depths are exact");
    require(fork_row.first_child[0] == 1 && fork_row.next_sibling[1] == 4 && fork_row.sibling_index[4] == 1,
            "root-fork child/sibling links are exact");
    require(fork_row.path_length[0] == 4 && fork_row.owned_from[0] == 0 &&
                fork_row.path_length[1] == 2 && fork_row.owned_from[1] == 1 &&
                fork_row.path_columns[1][0] == 0 && fork_row.path_columns[1][1] == 4,
            "root-fork paths/owned_from are exact");

    SpeculativeTreeRow chain_row{};
    build_root_fork_tree_row(chain_row, /*main_depth=*/3, /*has_side=*/false);
    require(chain_row.tree == 0 && chain_row.nodes == 4 && chain_row.paths == 1,
            "non-fork row builds chain tree metadata");

    // floor_verify_width maps all widths in [1, 32] below max_width down to supported buckets.
    for (std::int32_t w = 2; w <= 32; ++w) {
        const std::int32_t f = floor_verify_width(w, 32);
        require(f >= 2 && f <= w && f == bucket_verify_width(f, 32),
                "floor_verify_width returns a valid startup-captured bucket <= w");
    }
    require(floor_verify_width(11, 32) == 10 && floor_verify_width(13, 16) == 12 &&
                floor_verify_width(17, 16) == 16 && floor_verify_width(9, 16) == 8,
            "floor_verify_width handles exact boundary buckets");
    (void)kSpeculativeTreeMaxNodes;
    (void)kSpeculativeTreeMaxPathLength;
}

void check_concurrent_staged_width_bucketing() {
    using infernix::models::qwen4_exp::bucket_verify_width;
    using infernix::models::qwen4_exp::effective_staged_columns;

    constexpr std::int32_t kMaxWidth = 32;

    // For width <= 8, bucket_verify_width is identity so all rows retain staged columns.
    for (std::int32_t width = 2; width <= 8; ++width) {
        const std::int32_t W = bucket_verify_width(width, kMaxWidth);
        require(W == width, "width <= 8 is unbucketed");
        for (std::int32_t row = 0; row < 4; ++row) {
            require(effective_staged_columns(1, row, width, W, false) == 1,
                    "unbucketed neural row preserves anchor column");
            require(effective_staged_columns(width, row, width, W, false) == width,
                    "unbucketed ngram row preserves all staged columns");
        }
    }

    // For width > 8 that widens (W > width), row 0 keeps staged columns while rows > 0 reset to 0.
    for (const std::int32_t width : {9, 11, 13, 15, 17, 21, 25, 29, 31}) {
        const std::int32_t W = bucket_verify_width(width, kMaxWidth);
        require(W > width, "non-bucket width > 8 widens");
        require(effective_staged_columns(1, 0, width, W, false) == 1,
                "widened round preserves row 0 neural anchor");
        require(effective_staged_columns(width, 0, width, W, false) == width,
                "widened round preserves row 0 ngram columns");
        require(effective_staged_columns(1, 0, width, W, true) == 1,
                "widened tree round clamps row 0 to 1");
        require(effective_staged_columns(width, 0, width, W, true) == 1,
                "widened tree round clamps row 0 ngram to 1");
        for (std::int32_t row = 1; row < 4; ++row) {
            require(effective_staged_columns(1, row, width, W, false) == 0,
                    "widened round resets row > 0 anchor staging to prevent stride leak");
            require(effective_staged_columns(width, row, width, W, false) == 0,
                    "widened round resets row > 0 ngram staging to prevent stride leak");
            require(effective_staged_columns(width, row, width, W, true) == 0,
                    "widened tree round resets row > 0 staging");
        }
    }

    // Exact bucket widths keep W == width so rows > 0 retain staged columns.
    for (const std::int32_t width : {10, 12, 16, 20, 24, 28, 32}) {
        const std::int32_t W = bucket_verify_width(width, kMaxWidth);
        require(W == width, "exact bucket width does not widen");
        for (std::int32_t row = 0; row < 4; ++row) {
            require(effective_staged_columns(1, row, width, W, false) == 1,
                    "exact bucket preserves neural anchor");
            require(effective_staged_columns(width, row, width, W, false) == width,
                    "exact bucket preserves ngram columns");
        }
    }

    // End-to-end stride simulation: batch = 2, pre-staged at width = 11 (row 0 neural anchor, row 1
    // ngram copy), then verified at bucketed W = 12. Verify every slot (b, j) holds its own tag
    // without stale or cross-row data.
    constexpr std::int32_t kBatch = 2;
    constexpr std::int32_t kWidth = 11;
    const std::int32_t kBucketW   = bucket_verify_width(kWidth, kMaxWidth);
    require(kBucketW == 12, "width 11 buckets to 12");

    const auto tag = [](std::int32_t b, std::int32_t j) { return (b + 1) * 1000 + j; };
    std::vector<std::int32_t> slots(static_cast<std::size_t>(kBatch * kBucketW), -1);
    const std::array<std::int32_t, kBatch> staged{1, kWidth};

    // Stage pre-verification at stride kWidth.
    for (std::int32_t b = 0; b < kBatch; ++b) {
        for (std::int32_t j = 0; j < staged[b]; ++j) {
            slots[static_cast<std::size_t>(b * kWidth + j)] = tag(b, j);
        }
    }
    // Complete staging for verification at stride kBucketW starting from effective_staged_columns.
    for (std::int32_t b = 0; b < kBatch; ++b) {
        const std::int32_t from = effective_staged_columns(staged[b], b, kWidth, kBucketW, false);
        for (std::int32_t j = from; j < kBucketW; ++j) {
            slots[static_cast<std::size_t>(b * kBucketW + j)] = tag(b, j);
        }
    }
    for (std::int32_t b = 0; b < kBatch; ++b) {
        for (std::int32_t j = 0; j < kBucketW; ++j) {
            const std::int32_t got = slots[static_cast<std::size_t>(b * kBucketW + j)];
            require(got == tag(b, j),
                    "concurrent widened verify slot (" + std::to_string(b) + ", " + std::to_string(j) +
                        ") must not leak across rows or keep stale staging");
        }
    }

    // Upfront-bucketed pre-staging (staged_width == W == 12): row 1 preserves all 12 pre-staged
    // columns (zero re-staging); if post-draft stitching widens W further (12 -> 16), row 1 resets.
    require(effective_staged_columns(kBucketW, 1, kBucketW, kBucketW, false) == kBucketW,
            "upfront-bucketed pre-staging preserves row > 0 columns when W == staged_width");
    const std::int32_t kStitchedW = bucket_verify_width(13, kMaxWidth);
    require(kStitchedW == 16, "stitched width 13 buckets to 16");
    require(effective_staged_columns(kBucketW, 0, kBucketW, kStitchedW, false) == kBucketW,
            "post-draft stitching preserves row 0 columns");
    require(effective_staged_columns(kBucketW, 1, kBucketW, kStitchedW, false) == 0,
            "post-draft stitching resets row > 0 columns when W > staged_width");
}

void check_dcut_batch_pruning() {
    using infernix::models::qwen4_exp::calibrate_dcut_rate;
    using infernix::models::qwen4_exp::choose_dcut_batch_cap;
    using infernix::models::qwen4_exp::kMtpSampledCumulativeBins;
    using infernix::models::qwen4_exp::select_ngram_proposal;

    const auto multirow_cost = [](std::int32_t w) noexcept {
        return w <= 0 ? 1.0 : (1.0 + 0.05 * static_cast<double>(w));
    };

    // 1. batch == 1 must never prune below its single-row extent.
    {
        const std::array<std::int32_t, 1> ext{5};
        const std::array<std::array<double, 8>, 1> rates{{{{0.40, 0.35, 0.30, 0.25, 0.20, 0.0, 0.0, 0.0}}}};
        require(choose_dcut_batch_cap(1, ext, rates, multirow_cost) == 5,
                "D-Cut leaves batch=1 single-row extent unchanged");
    }

    // 2. Homogeneous high-confidence batch=4 keeps full extent 5.
    {
        const std::array<std::int32_t, 4> ext{5, 5, 5, 5};
        const std::array<double, 8> high{0.96, 0.96, 0.96, 0.96, 0.96, 0.0, 0.0, 0.0};
        const std::array<std::array<double, 8>, 4> rates{high, high, high, high};
        require(choose_dcut_batch_cap(4, ext, rates, multirow_cost) == 5,
                "D-Cut retains full width for homogeneous high-confidence batch=4");
    }

    // 3. Heterogeneous batch=4: one long lukewarm outlier (extent 5) while 3 rows stopped at 1
    // gets pruned so verification width does not balloon for a single low-yield tail.
    {
        const std::array<std::int32_t, 4> ext{5, 1, 1, 1};
        const std::array<std::array<double, 8>, 4> rates{{
            {0.66, 0.59, 0.51, 0.43, 0.32, 0.0, 0.0, 0.0},
            {0.51, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.51, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.51, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        }};
        const std::int32_t cap = choose_dcut_batch_cap(4, ext, rates, multirow_cost);
        require(cap >= 1 && cap < 3, "D-Cut prunes lukewarm outlier tail on heterogeneous batch=4");
    }

    // 4. All low-confidence neural rows: if zero marginal gain clears cost, prune down cleanly.
    {
        const std::array<std::int32_t, 4> ext{3, 3, 2, 2};
        const std::array<double, 8> poor{0.10, 0.10, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0};
        const std::array<std::array<double, 8>, 4> rates{poor, poor, poor, poor};
        require(choose_dcut_batch_cap(4, ext, rates, multirow_cost) <= 1,
                "D-Cut prunes low-survival batch=4 chain");
    }

    // 5. Mixed copy+neural batch: an n-gram copy row already paying width=12 makes verification
    // up to 12 columns free for neural rows, so neural drafts up to 5 are not trimmed.
    {
        const std::array<std::int32_t, 4> ext{5, 1, 0, 0}; // row 2 & 3 are n-gram copy rows (0 in dcut_extents)
        const std::array<std::array<double, 8>, 4> rates{{
            {0.51, 0.51, 0.43, 0.32, 0.26, 0.0, 0.0, 0.0},
            {0.51, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0},
            {0.0},
        }};
        const auto copy_cost_fn = [](std::int32_t w) noexcept {
            const std::int32_t vw = std::max(12, w);
            return 1.0 + 0.05 * static_cast<double>(vw);
        };
        require(choose_dcut_batch_cap(4, ext, rates, copy_cost_fn) == 5,
                "D-Cut does not trim neural drafts when an n-gram copy already sets wider batch width");
    }

    // 6. Greedy invariance: sampled observed/expected scaling and EMA blending do not alter greedy rates.
    {
        require(std::abs(calibrate_dcut_rate(0.84F, 1.0, 5.0, 0.20, false) - static_cast<double>(0.84F)) < 1e-12,
                "calibrate_dcut_rate leaves greedy rate unchanged regardless of sampled history");
    }

    // 7. Sampled online calibration triggers pruning where optimistic raw bin priors do not.
    {
        const std::array<std::int32_t, 4> ext{5, 5, 5, 5};
        const std::array<float, 5> raw_bins{
            kMtpSampledCumulativeBins.rates[6], // 0.91
            kMtpSampledCumulativeBins.rates[5], // 0.84
            kMtpSampledCumulativeBins.rates[5], // 0.84
            kMtpSampledCumulativeBins.rates[4], // 0.76
            kMtpSampledCumulativeBins.rates[4], // 0.76
        };
        std::array<std::array<double, 8>, 4> raw_rates{};
        std::array<std::array<double, 8>, 4> cal_rates{};
        const double observed = 2.2, expected = 5.0;
        const std::array<double, 5> ema{0.70, 0.50, 0.35, 0.25, 0.20};
        for (std::int32_t b = 0; b < 4; ++b) {
            for (std::int32_t j = 0; j < ext[b]; ++j) {
                raw_rates[b][static_cast<std::size_t>(j)] = static_cast<double>(raw_bins[static_cast<std::size_t>(j)]);
                cal_rates[b][static_cast<std::size_t>(j)] =
                    calibrate_dcut_rate(raw_bins[static_cast<std::size_t>(j)], observed, expected,
                                        ema[static_cast<std::size_t>(j)], true);
            }
        }
        const std::int32_t raw_cap = choose_dcut_batch_cap(4, ext, raw_rates, multirow_cost);
        const std::int32_t cal_cap = choose_dcut_batch_cap(4, ext, cal_rates, multirow_cost);
        require(raw_cap == 5, "raw sampled bin priors over-predict survival and fail to prune tail");
        require(cal_cap < 5, "online-calibrated sampled rates trigger D-Cut pruning on low-survival tail");
    }

    // 8. DFlash2-style calibrated survival pruning with piecewise cost and n-gram floor.
    {
        const std::array<std::int32_t, 4> ext{6, 6, 3, 2};
        const std::array<std::array<double, 8>, 4> df_rates{{
            {0.85, 0.65, 0.50, 0.35, 0.25, 0.15, 0.0, 0.0},
            {0.80, 0.60, 0.45, 0.30, 0.20, 0.10, 0.0, 0.0},
            {0.75, 0.55, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.70, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        }};
        const auto df_cost_none = [](std::int32_t w) noexcept {
            const double vc = w <= 0 ? 1.0 : (w <= 3 ? 1.0 + 0.05 * w : 1.15 + 0.05 * (w - 3));
            return vc + 0.08;
        };
        const std::int32_t pruned = choose_dcut_batch_cap(4, ext, df_rates, df_cost_none);
        require(pruned >= 2 && pruned < 6,
                "DFlash2-style calibrated rates prune low-yield trailing steps across batch=4");

        // When an n-gram copy sets width floor=4, cost is flat up to w=4 so cap >= 4.
        const auto df_cost_ngram4 = [](std::int32_t w) noexcept {
            const std::int32_t vw = std::max(4, w);
            return 1.0 + (vw <= 3 ? 0.05 * vw : 0.15 + 0.05 * (vw - 3)) + 0.08;
        };
        const std::array<std::int32_t, 4> ext_with_copy{6, 3, 0, 2};
        require(choose_dcut_batch_cap(4, ext_with_copy, df_rates, df_cost_ngram4) >= 4,
                "DFlash2-style D-Cut honors n-gram copy floor in batch round cost");
    }
}

void check_cross_request_suffix_proposal() {
    using infernix::models::qwen3_5::NgramMatch;
    using infernix::models::qwen3_5::detail::NgramProposer;
    using infernix::models::qwen4_exp::select_ngram_proposal;

    NgramProposer cross(256, 128);
    cross.set_boundaries({999});

    // Request 1 & 2 outputs ingested separately with boundary isolation.
    std::vector<std::int32_t> req1{10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    std::vector<std::int32_t> req2{200, 201, 202, 203, 204, 205, 999, 300, 301};
    cross.ingest(req1);
    cross.ingest(req2);

    // Cold request (empty in-request proposer) matches prior Request 1 via cross_request_proposer.
    NgramProposer cold_req(128, 64);
    const std::vector<std::int32_t> prompt_tail{10, 20, 30, 40, 50, 60};
    const auto local_miss = cold_req.propose(prompt_tail, 8, 6);
    require(local_miss.tokens.empty(), "cold request has no in-request match");
    const auto cross_hit = cross.propose(prompt_tail, 8, 6);
    const auto picked    = select_ngram_proposal(local_miss, cross_hit);
    require(picked.matched == 6 && picked.tokens == std::vector<std::int32_t>({70, 80, 90, 100}),
            "cross-request proposer rescues cold request without crossing boundaries");

    // Request 2 boundary token 999 prevents leaking across request/turn boundaries.
    const auto bleed_check = cross.propose(std::vector<std::int32_t>{200, 201, 202, 203, 204, 205}, 8, 6);
    require(bleed_check.tokens.empty(), "match never crosses boundary token or unrelated request tail");

    // In-request match takes precedence over equal or shorter cross-request match.
    NgramMatch local_strong{.tokens = {1, 2, 3}, .matched = 8};
    NgramMatch cross_shorter{.tokens = {4, 5, 6, 7}, .matched = 6};
    const auto prefer_local = select_ngram_proposal(local_strong, cross_shorter);
    require(prefer_local.matched == 8 && prefer_local.tokens == std::vector<std::int32_t>({1, 2, 3}),
            "longer in-request match overrides shorter cross-request match");

    // Compaction stress: ingest > 3x capacity (256) requests, newest pattern remains intact.
    for (int r = 0; r < 24; ++r) {
        std::vector<std::int32_t> fill(40);
        std::iota(fill.begin(), fill.end(), 1000 + r * 50);
        cross.ingest(fill);
    }
    const std::vector<std::int32_t> recent{2150, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159};
    cross.ingest(recent);
    const auto post_compact = cross.propose(std::vector<std::int32_t>{2150, 2151, 2152, 2153, 2154, 2155}, 8, 6);
    require(post_compact.matched == 6 &&
                post_compact.tokens == std::vector<std::int32_t>({2156, 2157, 2158, 2159}),
            "cross-request proposer remains accurate after sliding window compaction");
}

} // namespace

int main() {
    try {
        check_table();
        check_continuation_cut();
        check_min_logprob_cut();
        check_strided_rows();
        check_against_device_chain();
        check_copy_fallback();
        check_expected_tokens();
        check_sampled_cumulative_table();
        check_moe_round_cost_and_fit();
        check_ngram_buckets_and_moe_copy_stitch();
        check_root_fork_trees();
        check_concurrent_staged_width_bucketing();
        check_dcut_batch_pruning();
        check_cross_request_suffix_proposal();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("mtp round gating checks passed\n");
    return 0;
}
