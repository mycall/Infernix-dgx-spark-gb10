#pragma once

// The host side of a Qwen4Exp MTP draft round: which of a drafted chain the round verifies, how many
// draft steps the device actually ran, and whether an n-gram copy proposal is worth more per round
// time than the neural chain. Pure policy over the drafter's stored log-probabilities and the lane's
// learned per-position acceptances, so it is testable without a model.
//
// The continuation gate and the verify cut are two different decisions, and the token the gate stops
// on is not verified. The device chain (core/decode_graph.h's launch_graph_gate_product) stops after
// step j when the cumulative product of steps 0..j's binned rates falls below the round's threshold,
// so step j ran and its token is sunk; verifying it looked free, since one more column of a verify
// round costs a fraction of a plain round on GB10 (the [16384, 2560] NVFP4 GEMV is 104.3 us at T = 1
// against 106.3 us at T = 4). Measured on the W4Dense NVFP4 artifact it is not (2026-11-01, T6): the
// cut-position draft is accepted ~1-2% of the time, not the 12-26% its bin claims — the gate cuts
// exactly where the drafter's probability is overconfident — while every cut round pays that column's
// share of the verify path and every kept-0 round stops decaying to a plain one. Four interleaved
// ABBA passes with retention on: cold decode -2.1% (tg128) to -3.6% (tg256), warm -1.7%/-2.8%; before
// the cut-position outcome was excluded from the draft-length learner, -9.1%. The cut therefore drops
// the step it lands on, as before; the executed count below still reports what the round paid for.

#include "core/decode_graph.h"
#include "infernix/ops/speculative_tree.h"
#include "models/qwen3_5/ngram.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>

namespace infernix::models::qwen4_exp {

// The cumulative cut of an argmax MTP chain (candidates §7 D8; Basalt draft_cum.hpp): the chain keeps
// draft j while the product over i <= j of the observed acceptance rate of draft i's probability bin
// stays at or above the request's threshold. The bins are Basalt's, as log-probability edges (the
// drafter's stored values, compared without exp, so host and device classify alike:
// core/decode_graph.h). The threshold starts at kMtpCumulativeBase and falls toward
// kMtpCumulativeFloor as the request's drafts are accepted more often than the table predicts
// (observed / expected accepted, decayed 0.97 per round, mapped linearly over 0.9..1.1). Draft-only:
// verification and outputs are unchanged. It is the cut of `--draft-tokens auto` (Dense8 MTP auto,
// ABBA x5: conv 1.0529x, 5/5, worst 1.0296; struct 1.0009x); a fixed draft length keeps the
// --spec-min-p cut (MTP-3 under this cut: conv 1.0423x, struct 0.9919x, worse in every pair;
// 2026-10-28).
inline constexpr GraphGateBins kMtpCumulativeBins{
    {-1.20397280F, -0.693147181F, -0.356674944F, -0.223143551F, -0.105360516F, -0.0512932944F, -0.0100503359F},
    {0.12F, 0.26F, 0.43F, 0.51F, 0.59F, 0.66F, 0.79F, 0.96F}};
inline constexpr float kMtpCumulativeBase  = 0.40F;
inline constexpr float kMtpCumulativeFloor = 0.32F;

// The cumulative table for sampled 16-rank MTP chains (temperature > 0): because the drafter draws
// across up to 16 ranks and verification preserves target mass over that candidate set (block
// verification), empirical single-step acceptance exceeds the argmax top-1 bin probability in lower
// top-1 log-probability bands.
inline constexpr GraphGateBins kMtpSampledCumulativeBins{
    {-2.59026717F, -2.12026354F, -1.60943791F, -1.20397280F, -0.693147181F, -0.356674944F, -0.105360516F},
    {0.22F, 0.44F, 0.58F, 0.68F, 0.76F, 0.84F, 0.91F, 0.97F}};
inline constexpr float kMtpSampledCumulativeBase  = 0.28F;
inline constexpr float kMtpSampledCumulativeFloor = 0.20F;

struct MtpCumulative {
    float threshold = kMtpCumulativeBase;
    double observed = 0.0, expected = 0.0;

    void reset(float base = kMtpCumulativeBase) {
        *this     = MtpCumulative{};
        threshold = base;
    }
    // One verified chain: `offered` drafts with these log-probabilities, `accepted` of them taken.
    void observe(const float* logprobs, std::int32_t offered, std::int32_t accepted,
                 const GraphGateBins& bins = kMtpCumulativeBins,
                 float base = kMtpCumulativeBase, float floor = kMtpCumulativeFloor) {
        if (offered <= 0) { return; }
        constexpr double kDecay = 0.97, kLowRatio = 0.9, kHighRatio = 1.1;
        double chance = 1.0, sum = 0.0;
        for (std::int32_t j = 0; j < offered; ++j) {
            chance *= static_cast<double>(graph_gate_rate(logprobs[j], bins));
            sum += chance;
        }
        observed = observed * kDecay + static_cast<double>(accepted);
        expected = expected * kDecay + sum;
        if (expected < 0.5) { return; }
        const double t = std::clamp((observed / expected - kLowRatio) / (kHighRatio - kLowRatio), 0.0, 1.0);
        threshold      = base + static_cast<float>(t) * (floor - base);
    }
};

// A draft round's readout: the drafts the round verifies and the draft steps the device ran. They
// differ whenever a gated chain stopped early — the cut step ran but is not verified (header note),
// and the drafts beyond `executed` were never written and must not be read.
struct MtpDraftCut {
    std::int32_t kept     = 0;
    std::int32_t executed = 0;
};

// The cumulative cut of one row's chain: `steps` drafts at logprobs[j * stride], cut at the first j
// whose product over i <= j falls below `threshold` — the device gate's comparison and its binary32
// multiplications in the same order, so host and device agree on every chain. `gated` says whether the
// device ran the chain under that gate (mtp_draft_gated): there step j ran exactly when every earlier
// gate passed, so the chain's last executed step is the one it was cut on; an ungated chain ran every
// step and only the host cuts. The cut step itself is not verified even though its cost is sunk: see
// the header note for the GB10 measurement.
[[nodiscard]] inline MtpDraftCut mtp_cumulative_cut(const float* logprobs, std::int32_t stride, std::int32_t steps,
                                                    float threshold, const GraphGateBins& bins, bool gated) noexcept {
    MtpDraftCut cut{steps, steps};
    for (std::int32_t j = 0; j < steps; ++j) {
        const float product = graph_gate_product(logprobs, stride, j, bins);
        if (product >= threshold) { continue; }
        cut.executed = gated ? j + 1 : steps;
        cut.kept     = j;
        break;
    }
    return cut;
}

// The fixed-length cut of one row's chain (--spec-min-p): cut at the first draft below
// `min_logprob`, which is also the device gate's comparison. There the threshold is an explicit floor
// on what a round may verify, not only on what it may keep drafting; the executed count is still
// reported, as the round paid for the cut step either way.
[[nodiscard]] inline MtpDraftCut mtp_min_logprob_cut(const float* logprobs, std::int32_t stride, std::int32_t steps,
                                                     float min_logprob, bool gated) noexcept {
    MtpDraftCut cut{steps, steps};
    for (std::int32_t j = 0; j < steps; ++j) {
        if (logprobs[j * stride] >= min_logprob) { continue; }
        cut.executed = gated ? j + 1 : steps;
        cut.kept     = j;
        break;
    }
    return cut;
}

// Expected tokens a round commits by verifying `count` drafts whose per-position acceptance is
// `accept`: the target's own token, then each draft whose whole prefix was accepted.
[[nodiscard]] inline double expected_copy_tokens(std::int32_t count, double accept) noexcept {
    double reach = 1.0, tokens = 1.0;
    for (std::int32_t j = 0; j < count; ++j) {
        reach *= accept;
        tokens += reach;
    }
    return tokens;
}

// The prefix length in [0, copies] that maximizes expected tokens per round time under `copy_accept`
// and `copy_cost`, while enforcing an optional minimum expected per-draft acceptance rate
// `min_accept` (sum_{j=1..k} copy_accept^j / k >= min_accept). Prevents a long --ngram-draft-tokens
// limit from diluting a match's score or drafting low-probability tails that drag mixed acceptance
// below target.
[[nodiscard]] inline std::int32_t best_copy_length(std::int32_t copies, double copy_accept,
                                                   double copy_cost,
                                                   double min_accept = 0.0) noexcept {
    std::int32_t best = 0;
    double best_score = 1.0;
    double reach = 1.0, accepted = 0.0;
    for (std::int32_t k = 1; k <= copies; ++k) {
        reach *= copy_accept;
        accepted += reach;
        if (min_accept > 0.0 && accepted < min_accept * static_cast<double>(k)) { break; }
        const double score = (1.0 + accepted) / (1.0 + copy_cost * static_cast<double>(k));
        if (score > best_score) {
            best_score = score;
            best       = k;
        }
    }
    return best;
}

// The same for a neural chain: `drafts` drafts with the lane's learned per-position acceptances
// accept[0..drafts).
[[nodiscard]] inline double expected_chain_tokens(std::span<const double> accept, std::int32_t drafts) noexcept {
    double reach = 1.0, tokens = 1.0;
    const auto limit = std::min(drafts, static_cast<std::int32_t>(accept.size()));
    for (std::int32_t j = 0; j < limit; ++j) {
        reach *= accept[static_cast<std::size_t>(j)];
        tokens += reach;
    }
    return tokens;
}

// Whether a round verifies an n-gram copy rather than a neural chain: expected tokens per round time,
// the copy's columns costing `copy_cost` and the chain's drafts `chain_cost` plain rounds each (a copy
// needs no draft step). The chain may be shorter than the drafts the lane drafted — after the round's
// cut, only its kept prefix still counts.
[[nodiscard]] inline bool copy_beats_chain(std::int32_t copies, double copy_accept, double copy_cost,
                                           std::span<const double> chain_accept, std::int32_t chain_drafts,
                                           double chain_cost) noexcept {
    const double copy_score =
        expected_copy_tokens(copies, copy_accept) / (1.0 + copy_cost * static_cast<double>(copies));
    const double chain_score =
        expected_chain_tokens(chain_accept, chain_drafts) / (1.0 + chain_cost * static_cast<double>(chain_drafts));
    return copy_score > chain_score;
}

// Expected distinct routed experts across `columns` tokens when each token routes uniformly to
// `top_k` of `experts` (coupon-collector model): E * (1 - (1 - K / E)^columns).
[[nodiscard]] inline double moe_expected_unique_experts(std::int32_t experts, std::int32_t top_k,
                                                        std::int32_t columns) noexcept {
    if (columns <= 0 || experts <= 0 || top_k <= 0) { return 0.0; }
    if (top_k >= experts) { return static_cast<double>(experts); }
    const double e = static_cast<double>(experts);
    const double k = static_cast<double>(top_k);
    return e * (1.0 - std::pow(1.0 - k / e, static_cast<double>(columns)));
}

// Relative round time of a speculative round verifying `drafts` per sequence across `batch`
// sequences compared to a plain round (drafts = 0) of the same `batch`, modeling dense layers as
// `dense_equiv_experts` expert-equivalents plus the expected distinct routed experts plus `drafts`
// serial draft steps normalized from B=1 plain rounds by (dense + U(1)) / (dense + U(batch)).
[[nodiscard]] inline double moe_round_cost(std::int32_t batch, std::int32_t drafts, std::int32_t experts,
                                           std::int32_t top_k, double dense_equiv_experts,
                                           double draft_step_cost = 0.0) noexcept {
    if (drafts <= 0) { return 1.0; }
    const double u_b1         = moe_expected_unique_experts(experts, top_k, 1);
    const double u0           = moe_expected_unique_experts(experts, top_k, std::max(batch, 1));
    const double u1           = moe_expected_unique_experts(experts, top_k, std::max(batch, 1) * (1 + drafts));
    const double denom        = std::max(dense_equiv_experts + u0, 1e-6);
    const double verify_ratio = (dense_equiv_experts + u1) / denom;
    const double step_scale   = (dense_equiv_experts + u_b1) / denom;
    return verify_ratio + draft_step_cost * step_scale * static_cast<double>(drafts);
}

// Online Ridge-regularized EWM linear regression of single-row decode round duration:
//   t = a + b_v * x_v + b_d * x_d
// where x_v = (U(1 + verified_drafts) - U(1)) / U(1) is normalized extra unique routed experts
// verified across all B=1 round kinds (plain, complete MTP, gated MTP, N-gram), and x_d is the
// executed draft step count (0 for plain / pure N-gram, executed steps for MTP).
struct RoundCostModel {
    static constexpr double kDecay       = 0.98;
    static constexpr double kGapRatio    = 3.0;
    static constexpr double kMinRounds   = 16.0;
    static constexpr double kRidgeLambda = 4.0;

    double n     = 0.0;
    double t_sum = 0.0;
    double m[3][3]{};
    double rhs[3]{};

    [[nodiscard]] static double extra_expert_feature(std::int32_t verified_drafts, std::int32_t experts,
                                                     std::int32_t top_k) noexcept {
        if (verified_drafts <= 0) { return 0.0; }
        const double u1 = moe_expected_unique_experts(experts, top_k, 1);
        const double uk = moe_expected_unique_experts(experts, top_k, 1 + verified_drafts);
        return (uk - u1) / std::max(u1, 1e-6);
    }

    [[nodiscard]] bool solve_ridge(std::int32_t experts, std::int32_t top_k, double prior_dense_equiv,
                                   double prior_step_cost, double& a, double& bv, double& bd) const noexcept {
        if (n < 1.0 || t_sum <= 0.0) { return false; }
        const double u1       = std::max(moe_expected_unique_experts(experts, top_k, 1), 1.0);
        const double ratio_v0 = u1 / std::max(prior_dense_equiv + u1, 1.0);
        // Scale-invariant penalty kRidgeLambda * ((bv - ratio_v0 * a)^2 + (bd - prior_step_cost * a)^2)
        // keeps the intercept `a` unbiased while shrinking normalized cost ratios toward priors.
        const std::array<double, 3> v1{-ratio_v0, 1.0, 0.0};
        const std::array<double, 3> v2{-prior_step_cost, 0.0, 1.0};

        double aug[3][4]{};
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                aug[r][c] = m[r][c] +
                            kRidgeLambda * (v1[static_cast<std::size_t>(r)] * v1[static_cast<std::size_t>(c)] +
                                            v2[static_cast<std::size_t>(r)] * v2[static_cast<std::size_t>(c)]);
            }
            aug[r][3] = rhs[r];
        }
        for (int col = 0; col < 3; ++col) {
            int piv = col;
            double best = std::abs(aug[col][col]);
            for (int r = col + 1; r < 3; ++r) {
                if (std::abs(aug[r][col]) > best) {
                    best = std::abs(aug[r][col]);
                    piv  = r;
                }
            }
            if (best < 1e-18) { return false; }
            if (piv != col) {
                for (int c = col; c < 4; ++c) { std::swap(aug[col][c], aug[piv][c]); }
            }
            const double inv = 1.0 / aug[col][col];
            for (int r = col + 1; r < 3; ++r) {
                const double factor = aug[r][col] * inv;
                for (int c = col; c < 4; ++c) { aug[r][c] -= factor * aug[col][c]; }
            }
        }
        double x[3]{};
        for (int r = 2; r >= 0; --r) {
            double s = aug[r][3];
            for (int c = r + 1; c < 3; ++c) { s -= aug[r][c] * x[c]; }
            x[r] = s / aug[r][r];
        }
        a  = x[0];
        bv = x[1];
        bd = x[2];
        return std::isfinite(a) && std::isfinite(bv) && std::isfinite(bd) && a > 0.0;
    }

    void observe(std::int32_t verified_drafts, std::int32_t executed_draft_steps, double seconds,
                 std::int32_t experts, std::int32_t top_k, double prior_dense_equiv,
                 double prior_step_cost) noexcept {
        if (!(seconds > 0.0) || !std::isfinite(seconds) || verified_drafts < 0 || executed_draft_steps < 0) {
            return;
        }
        const double xv = extra_expert_feature(verified_drafts, experts, top_k);
        const double xd = static_cast<double>(executed_draft_steps);
        double a = 0.0, bv = 0.0, bd = 0.0;
        if (n >= 4.0) {
            const double u1 = std::max(moe_expected_unique_experts(experts, top_k, 1), 1.0);
            const double predict = solve_ridge(experts, top_k, prior_dense_equiv, prior_step_cost, a, bv, bd)
                                       ? (a + std::max(0.0, bv) * xv + std::max(0.0, bd) * xd)
                                       : (t_sum / n) * moe_round_cost(1, verified_drafts, experts, top_k,
                                                                      prior_dense_equiv, prior_step_cost) /
                                             std::max(moe_round_cost(1, 0, experts, top_k, prior_dense_equiv, 0.0), 1.0);
            if (predict > 0.0 && seconds > kGapRatio * predict) { return; }
        }
        const double f[3] = {1.0, xv, xd};
        n                 = kDecay * n + 1.0;
        t_sum             = kDecay * t_sum + seconds;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) { m[r][c] = kDecay * m[r][c] + f[r] * f[c]; }
            rhs[r] = kDecay * rhs[r] + f[r] * seconds;
        }
    }

    [[nodiscard]] bool fit(std::int32_t experts, std::int32_t top_k, double prior_dense_equiv,
                           double prior_step_cost, double& out_dense_equiv,
                           double& out_step_cost) const noexcept {
        if (n < kMinRounds) { return false; }
        double a = 0.0, bv = 0.0, bd = 0.0;
        if (!solve_ridge(experts, top_k, prior_dense_equiv, prior_step_cost, a, bv, bd)) { return false; }
        const double u1 = std::max(moe_expected_unique_experts(experts, top_k, 1), 1.0);
        out_dense_equiv =
            bv > 0.0 ? std::clamp(u1 * std::max(a / bv - 1.0, 0.25), 35.0, 300.0) : prior_dense_equiv;
        out_step_cost = std::clamp(bd / a, 0.02, 0.12);
        return true;
    }
};

// Prefix length in [0, copies] maximizing expected tokens / moe_round_cost while preserving the
// minimum expected per-draft acceptance rate `min_accept`.
[[nodiscard]] inline std::int32_t best_copy_length_moe(std::int32_t copies, double copy_accept, std::int32_t batch,
                                                       std::int32_t experts, std::int32_t top_k, double dense_equiv,
                                                       double min_accept = 0.0) noexcept {
    std::int32_t best = 0;
    double best_score = 1.0;
    double reach = 1.0, accepted = 0.0;
    for (std::int32_t k = 1; k <= copies; ++k) {
        reach *= copy_accept;
        accepted += reach;
        if (min_accept > 0.0 && accepted < min_accept * static_cast<double>(k)) { break; }
        const double cost  = moe_round_cost(batch, k, experts, top_k, dense_equiv, 0.0);
        const double score = (1.0 + accepted) / cost;
        if (score > best_score) {
            best_score = score;
            best       = k;
        }
    }
    return best;
}

[[nodiscard]] inline bool copy_beats_chain_moe(std::int32_t copies, double copy_accept,
                                               std::span<const double> chain_accept, std::int32_t chain_drafts,
                                               std::int32_t batch, std::int32_t experts, std::int32_t top_k,
                                               double dense_equiv, double draft_step_cost) noexcept {
    const double copy_score =
        expected_copy_tokens(copies, copy_accept) / moe_round_cost(batch, copies, experts, top_k, dense_equiv, 0.0);
    const double chain_score =
        expected_chain_tokens(chain_accept, chain_drafts) /
        moe_round_cost(batch, chain_drafts, experts, top_k, dense_equiv, draft_step_cost);
    return copy_score > chain_score;
}

[[nodiscard]] inline constexpr std::int32_t bucket_verify_width(std::int32_t width,
                                                                std::int32_t max_width) noexcept {
    if (width <= 8 || width >= max_width) { return std::min(width, max_width); }
    constexpr std::int32_t kBuckets[] = {10, 12, 16, 20, 24, 28, 32};
    for (const std::int32_t b : kBuckets) {
        if (b >= width) { return std::min(b, max_width); }
    }
    return max_width;
}

// Columns already staged at stride `width` remain valid for row 0 when `bucketed_width != width`,
// but rows b > 0 were written at `b * width + j` instead of `b * bucketed_width + j` and must be
// re-staged from column 0 to prevent cross-row / stale PLE n-gram leaks.
[[nodiscard]] inline constexpr std::int32_t effective_staged_columns(
    std::int32_t staged, std::int32_t row, std::int32_t width, std::int32_t bucketed_width,
    bool tree_row) noexcept {
    const std::int32_t s = (bucketed_width != width && row > 0) ? 0 : staged;
    return tree_row ? std::min(s, 1) : std::min(s, bucketed_width);
}

[[nodiscard]] inline constexpr std::int32_t floor_verify_width(std::int32_t width,
                                                               std::int32_t max_width) noexcept {
    if (width <= 8) { return width; }
    const std::int32_t clamped = std::min(width, max_width);
    if (clamped == max_width) { return max_width; }
    constexpr std::int32_t kBuckets[] = {32, 28, 24, 20, 16, 12, 10, 8};
    for (const std::int32_t b : kBuckets) {
        if (b <= clamped) { return b; }
    }
    return 8;
}

[[nodiscard]] inline std::uint32_t count_bucketed_verify_widths(std::int32_t max_width) noexcept {
    std::uint32_t n = 0;
    for (std::int32_t w = 2; w <= max_width; ++w) {
        if (w == bucket_verify_width(w, max_width)) { ++n; }
    }
    return n;
}

// Eight-bucket confidence classifier combining SAM match length, occurrence multiplicity, recency,
// and neural confirmation / post-neural stitching:
//   0: short (<4, or 4..7 with occurrences < 3)
//   1: 8..11 single stale (or 4..7 with occurrences >= 3)
//   2: 8..11 repeated (>=2 occ) or recent (<=2048 tok) (or 6..7 with occurrences >= 6)
//   3: 12..15 single stale      4: 12..15 repeated or recent
//   5: 16+ single stale         6: 16+ repeated or recent
//   7: neural-confirmed or post-neural stitched SAM suffix
[[nodiscard]] inline int ngram_confidence_bucket(std::uint32_t matched, std::uint32_t occurrences,
                                                 std::uint32_t distance, bool mtp_confirmed = false) noexcept {
    if (mtp_confirmed) { return 7; }
    if (matched < 8U) {
        if (matched >= 4U && occurrences >= 3U) {
            return (matched >= 6U && occurrences >= 6U) ? 2 : 1;
        }
        return 0;
    }
    const bool strong = (occurrences >= 2U) || (distance <= 2048U);
    if (matched < 12U) { return strong ? 2 : 1; }
    if (matched < 16U) { return strong ? 4 : 3; }
    return strong ? 6 : 5;
}

// Optimal stitched SAM suffix length in [0, copies] appended after an already-drafted neural prefix
// of length `prefix_len` with expected round tokens `prefix_expected_tokens` and survival
// probability `prefix_reach` at its tip (draft step cost already sunk).
[[nodiscard]] inline std::int32_t best_stitch_length(std::int32_t prefix_len, double prefix_expected_tokens,
                                                     double prefix_reach, std::int32_t copies, double copy_accept,
                                                     std::int32_t batch, std::int32_t current_batch_drafts,
                                                     std::int32_t experts, std::int32_t top_k, double dense_equiv,
                                                     double min_accept = 0.0) noexcept {
    if (copies <= 0 || prefix_reach <= 0.0) { return 0; }
    std::int32_t best = 0;
    double best_score =
        prefix_expected_tokens / moe_round_cost(batch, current_batch_drafts, experts, top_k, dense_equiv, 0.0);
    double reach = 1.0, accepted = 0.0;
    for (std::int32_t s = 1; s <= copies; ++s) {
        reach *= copy_accept;
        accepted += reach;
        if (min_accept > 0.0 && accepted < min_accept * static_cast<double>(s)) { break; }
        const std::int32_t width_drafts = std::max(current_batch_drafts, prefix_len + s);
        const double total_tokens       = prefix_expected_tokens + prefix_reach * accepted;
        const double score =
            total_tokens / moe_round_cost(batch, width_drafts, experts, top_k, dense_equiv, 0.0);
        if (score > best_score) {
            best_score = score;
            best       = s;
        }
    }
    return best;
}

struct RootForkDecision {
    bool fork                = false;
    std::int32_t side_token  = 0;
};

// Decides independently of the realized draw `main_token` whether step-0 proposal uncertainty
// warrants a depth-1 root-fork branch, then samples an alternative token without replacement from
// q restricted to candidates != main_token using a deterministic hash of (seed, position).
// Gates out `(position % ratio) == (ratio - 2)` where QSA pooled-block completion would see both
// the main token and side token at physical position `position + 1`.
[[nodiscard]] inline RootForkDecision choose_root_fork_side(std::int32_t main_token,
                                                            std::span<const std::int32_t> candidates,
                                                            std::span<const float> q, std::int32_t position,
                                                            std::int32_t ratio, std::uint64_t seed,
                                                            float max_p_threshold = 0.90F) noexcept {
    if (ratio > 1 && (position % ratio) == (ratio - 2)) { return {}; }
    if (candidates.empty() || candidates.size() != q.size()) { return {}; }
    float max_p = 0.0F;
    for (const float p : q) {
        if (std::isfinite(p) && p > max_p) { max_p = p; }
    }
    if (max_p >= max_p_threshold) { return {}; }
    double residual = 0.0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i] != main_token && std::isfinite(q[i]) && q[i] > 0.0F) { residual += q[i]; }
    }
    if (residual <= 1e-6) { return {}; }
    std::uint64_t x = seed ^ (static_cast<std::uint64_t>(position) * 0x9E3779B97F4A7C15ULL);
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= (x >> 31);
    const double u       = static_cast<double>(x >> 11) * (1.0 / static_cast<double>(std::uint64_t{1} << 53));
    const double target  = u * residual;
    double accum         = 0.0;
    std::int32_t picked  = -1;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i] == main_token || !std::isfinite(q[i]) || q[i] <= 0.0F) { continue; }
        picked = candidates[i];
        accum += q[i];
        if (accum >= target) { return {.fork = true, .side_token = picked}; }
    }
    return picked >= 0 ? RootForkDecision{.fork = true, .side_token = picked} : RootForkDecision{};
}

// Builds a depth-1 root-fork SpeculativeTreeRow (transport = 0, without-replacement):
// path 0 is the main chain `0..main_depth`; when `has_side`, column `main_depth + 1` forks off
// root column 0 at depth 1 and path 1 is `{0, main_depth + 1}`.
inline void build_root_fork_tree_row(ops::SpeculativeTreeRow& row, std::int32_t main_depth, bool has_side) noexcept {
    std::memset(&row, 0, sizeof(row));
    std::fill(std::begin(row.parent), std::end(row.parent), static_cast<std::int8_t>(-1));
    std::fill(std::begin(row.first_child), std::end(row.first_child), static_cast<std::int8_t>(-1));
    std::fill(std::begin(row.next_sibling), std::end(row.next_sibling), static_cast<std::int8_t>(-1));
    row.main_depth = main_depth;
    row.tree       = has_side ? 1 : 0;
    row.transport  = 0;
    row.sampled    = 0;
    for (std::int32_t c = 1; c <= main_depth; ++c) {
        row.parent[c]          = static_cast<std::int8_t>(c - 1);
        row.depth[c]           = static_cast<std::int8_t>(c);
        row.first_child[c - 1] = static_cast<std::int8_t>(c);
        row.sibling_index[c]   = 0;
        row.path_columns[0][c] = static_cast<std::int8_t>(c);
    }
    row.path_length[0] = static_cast<std::int8_t>(main_depth + 1);
    row.owned_from[0]  = 0;
    if (has_side) {
        const std::int32_t s = main_depth + 1;
        row.nodes            = main_depth + 2;
        row.paths            = 2;
        row.parent[s]        = 0;
        row.depth[s]         = 1;
        row.next_sibling[1]  = static_cast<std::int8_t>(s);
        row.sibling_index[s] = 1;
        row.path_length[1]   = 2;
        row.owned_from[1]    = 1;
        row.path_columns[1][0] = 0;
        row.path_columns[1][1] = static_cast<std::int8_t>(s);
    } else {
        row.nodes = main_depth + 1;
        row.paths = 1;
    }
}

// Calibrates a binned per-step acceptance rate for D-Cut. Greedy chains use the bin table
// directly; sampled chains (T > 0) scale the bin prior by the lane's online observed/expected
// ratio from MtpCumulative and blend with the lane's per-position EMA acceptance so D-Cut
// prunes low-survival sampled tails instead of trusting optimistic static bin priors.
[[nodiscard]] inline double calibrate_dcut_rate(
    float bin_rate, double observed, double expected, double lane_pos_accept, bool sampled) noexcept {
    const double base = std::clamp(static_cast<double>(bin_rate), 0.0, 1.0);
    if (!sampled) { return base; }
    const double ratio  = expected >= 0.5 ? std::clamp(observed / expected, 0.25, 1.25) : 1.0;
    const double scaled = std::clamp(base * ratio, 0.0, 1.0);
    const double ema    = std::clamp(lane_pos_accept, 0.0, 1.0);
    return 0.5 * scaled + 0.5 * ema;
}

// D-Cut cross-request verification budget pruning: given per-row surviving neural extents
// (`row_extents[b]`, 0 for copy or inactive rows), per-step calibrated acceptance rates
// (`row_rates[b][j]`), and round verification cost `cost_fn(w)` at shared width `w`, return
// the shared cap `best_k` in `[0, max_ext]` maximizing batch-wide expected committed tokens
// per unit round time. Ties keep the smaller `w` to save verification compute.
template <class CostFn>
[[nodiscard]] inline std::int32_t choose_dcut_batch_cap(
    std::int32_t batch, std::span<const std::int32_t> row_extents,
    std::span<const std::array<double, 8>> row_rates, CostFn&& cost_fn) noexcept {
    if (batch <= 0 || row_extents.size() < static_cast<std::size_t>(batch) ||
        row_rates.size() < static_cast<std::size_t>(batch)) {
        return 0;
    }
    std::int32_t max_ext = 0;
    for (std::int32_t b = 0; b < batch; ++b) {
        max_ext = std::max(max_ext, std::clamp(row_extents[b], 0, 8));
    }
    if (batch <= 1 || max_ext <= 1) { return max_ext; }
    std::int32_t best = 0;
    double best_score = static_cast<double>(batch) / std::max(static_cast<double>(cost_fn(0)), 1e-6);
    for (std::int32_t w = 1; w <= max_ext; ++w) {
        double expected_tokens = 0.0;
        for (std::int32_t b = 0; b < batch; ++b) {
            expected_tokens += 1.0;
            const std::int32_t lim = std::min(std::clamp(row_extents[b], 0, 8), w);
            double reach           = 1.0;
            for (std::int32_t j = 0; j < lim; ++j) {
                reach *= std::clamp(row_rates[b][static_cast<std::size_t>(j)], 0.0, 1.0);
                expected_tokens += reach;
            }
        }
        const double score = expected_tokens / std::max(static_cast<double>(cost_fn(w)), 1e-6);
        if (score > best_score) {
            best_score = score;
            best       = w;
        }
    }
    return best;
}

// Selects between an in-request suffix match and a cross-request suffix match: in-request takes
// precedence unless empty or strictly outmatched in matched prefix length (or continuation
// length on ties).
[[nodiscard]] inline qwen3_5::NgramMatch select_ngram_proposal(
    qwen3_5::NgramMatch local, qwen3_5::NgramMatch cross) noexcept {
    if (local.tokens.empty()) { return cross; }
    if (!cross.tokens.empty() &&
        (cross.matched > local.matched ||
         (cross.matched == local.matched && cross.tokens.size() > local.tokens.size()))) {
        return cross;
    }
    return local;
}

} // namespace infernix::models::qwen4_exp
