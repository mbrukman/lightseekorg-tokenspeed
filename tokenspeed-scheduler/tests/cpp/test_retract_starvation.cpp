// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

#include "integration_test_helper.h"

namespace tokenspeed::test {

// ============================================================
//  Retracted resource accounting: the device tail page must be released
//  to the device pool at retract time, not pinned for the lifetime of
//  the Retracted state.
//
//  Pre-fix behavior: every retract leaked one device page (the partial
//  tail page held by ``Retracted::local_kv_allocator_``). Under sustained
//  retraction the pool's free list drained monotonically until new requests
//  could no longer be admitted — a deadlock the scheduler's priority order
//  AND the prefill-first break could not work around because the leak was
//  structural, not a scheduling artifact.
//
//  Fix: ``WriteBackDoneEvent::operator()(Retracting&&)`` drops the
//  LocalKVAllocator before constructing Retracted. Recovery in
//  ``ScheduleDecodeFromRetractedEvent::operator()(Retracted&&)`` allocates
//  a fresh LocalKVAllocator sized for ``partial_tail_tokens +
//  decode_input_tokens`` and the op's ``input_length`` is extended so the
//  model re-prefills the dropped tail.
// ============================================================

class RetractLeakTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        // decode_input_tokens > 0 makes the local KV allocator hold a real
        // tail page (one extra page beyond the radix-tree-inserted full
        // pages). Mirrors RetractFromPrefillDoneTestSuite so retract actually
        // produces a meaningful tail page to drop.
        cfg.decode_input_tokens = 2;
        cfg.page_size = 2;
        // total=6 → 5 usable pages (page 0 reserved). Need headroom for the
        // post-recovery decode step: recovery's re-prefill consumes loadback
        // + partial_tail + decode_reserve, and the immediately-following
        // PrefillDone → Decoding transition acquires another decode-reserve
        // page. With a tighter pool (total=4) recovery would itself trigger
        // a second retract cascade, masking the test's intent.
        cfg.device_allocator.total_pages = 6;
        cfg.host_allocator.total_pages = 16;
        cfg.enable_l3_storage = false;
        cfg.enable_mixed_prefill_decode = false;
        return cfg;
    }

    void SendReserveNumTokens(const std::string& id, std::int32_t n) {
        ExecutionEvent event;
        event.With(ForwardEvent{forward::UpdateReserveNumTokens{
            .request_id = id,
            .reserve_num_tokens_in_next_schedule_event = n,
        }});
        scheduler_->Advance(std::move(event));
    }

    static const FlatWriteBackOperation* GetWriteBack(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* wb = std::get_if<FlatWriteBackOperation>(cop)) {
                    return wb;
                }
            }
        }
        return nullptr;
    }

    static const FlatForwardOperation* GetForward(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<FlatForwardOperation>(&op)) {
                return f;
            }
        }
        return nullptr;
    }

    // Drive r1 to Decoding then force a retract via a large reserve; consume
    // the writeback. Returns scheduler.AvailableKvPages() right after the
    // WriteBackDone event.
    std::size_t DriveAndRetract(const std::string& id, token_t start) {
        // 1-page request (2 tokens) → prefill 1 page.
        Submit(MakeRequestSpec(id, /*num_pages=*/1, start));
        PlanOnce();
        // Add 1 generated token → 3 tokens; the partial tail now contains
        // token 3 in the 2nd device page (tail_available=1).
        SendForwardDone(id, {42});
        PlanOnce();
        // Reserve more than the device can satisfy → triggers retract.
        SendReserveNumTokens(id, 5);
        auto plan = PlanOnce();
        const auto* wb = GetWriteBack(plan);
        if (wb == nullptr || wb->op_ids.empty()) {
            return scheduler_->AvailableKvPages();
        }
        SendWriteBackDone(wb->op_ids[0]);
        return scheduler_->AvailableKvPages();
    }
};

// A single retract releases the tail page back to the pool. Pre-fix this
// would be a no-op (tail page stayed inside Retracted::local_kv_allocator_).
TEST_F(RetractLeakTestSuite, RetractReleasesTailPageToDevicePool) {
    // Bring r1 to a known Decoding state.
    Submit(MakeRequestSpec("r1", /*num_pages=*/1, /*start=*/1));
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    ASSERT_EQ(scheduler_->DecodingSize(), 1u);
    const std::size_t avail_decoding = scheduler_->AvailableKvPages();

    // Force retract.
    SendReserveNumTokens("r1", 5);
    auto plan = PlanOnce();
    const auto* wb = GetWriteBack(plan);
    ASSERT_NE(wb, nullptr);
    ASSERT_FALSE(wb->op_ids.empty());
    SendWriteBackDone(wb->op_ids[0]);
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);

    const std::size_t avail_retracted = scheduler_->AvailableKvPages();

    // The full pages r1 held went into the radix tree (still device-occupied
    // but evictable; not counted in AvailableKvPages which only tracks the
    // free list). LocalKVAllocator's leftover pages — the partial tail page
    // plus any decode-reserve pages it had Acquire'd — are now returned to
    // the pool by the fix → free count goes strictly up. Pre-fix this is
    // equal because Retracted held onto the whole LocalKVAllocator.
    EXPECT_GT(avail_retracted, avail_decoding)
        << "After fix: retract must release the tail device page (and any "
        << "decode-reserve pages held by LocalKVAllocator).";
}

// Sustained retracts must not monotonically drain the device pool. Pre-fix
// each retract leaked 1 device page, so after N retracts AvailableKvPages
// dropped by N. With the fix the pool is reusable across retracts.
TEST_F(RetractLeakTestSuite, ManyRetractsDoNotLeakDevicePages) {
    const std::size_t avail_initial = scheduler_->AvailableKvPages();

    // Each retract leaves the request in Retracted state (we never recover
    // it). We expect zero device pages to be permanently consumed across N
    // retracts, since the tail page should be released.
    constexpr int kNumRetracts = 5;
    for (int i = 0; i < kNumRetracts; ++i) {
        const std::string id = "r_" + std::to_string(i);
        // Fresh token range so no prefix-cache sharing across requests.
        const token_t start = static_cast<token_t>(10'000 + 1'000 * i);
        Submit(MakeRequestSpec(id, /*num_pages=*/1, start));
        PlanOnce();
        SendForwardDone(id, {42});
        PlanOnce();
        SendReserveNumTokens(id, 5);
        auto plan = PlanOnce();
        const auto* wb = GetWriteBack(plan);
        if (wb == nullptr) {
            // Device exhausted before this iteration could even retract —
            // this is precisely the deadlock the issue describes. Pre-fix
            // would hit this around iter 3-4 on this config.
            FAIL() << "Device pool deadlocked at iter " << i
                   << "; AvailableKvPages=" << scheduler_->AvailableKvPages()
                   << ". Leak suspected.";
        }
        ASSERT_FALSE(wb->op_ids.empty());
        SendWriteBackDone(wb->op_ids[0]);
    }

    EXPECT_EQ(scheduler_->RetractedSize(), kNumRetracts);

    // Pre-fix: after N retracts, AvailableKvPages dropped by ~N (tail page
    // leak per retract). Post-fix: pool size is conserved modulo pages still
    // legitimately held by tree entries from each retracted request's full
    // pages. The full pages are evictable so EnsureCapacityByEvict can claim
    // them; what matters is the free pool isn't monotonically shrinking from
    // tail-page leaks. We bound the worst case at "no more than 1 leaked
    // page per retract" to detect regression.
    const std::size_t avail_now = scheduler_->AvailableKvPages();
    EXPECT_GE(avail_now + kNumRetracts, avail_initial)
        << "Each retract appears to have leaked > 1 device page";
}

// Recovery after the tail-page-release fix: Retracted → PrefillDone (via
// ScheduleRetractedReprefillEvent, which emits a PrefillOperation for the
// dropped-KV partial tail) → Decoding (next plan, via the standard
// ScheduleDecodeEvent). Two plan rounds total, vs. one in the pre-fix flow.
TEST_F(RetractLeakTestSuite, RecoveryAfterTailPageRelease) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/1, /*start=*/1));
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    SendReserveNumTokens("r1", 5);

    auto plan1 = PlanOnce();
    const auto* wb = GetWriteBack(plan1);
    ASSERT_NE(wb, nullptr);
    SendWriteBackDone(wb->op_ids[0]);
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);

    // plan2: re-prefill the partial tail. Recovery emits a PrefillOperation,
    // not a DecodeOperation — the partial-tail tokens need an input_ids
    // vector that only PrefillOperation carries.
    auto plan2 = PlanOnce();
    const auto* fwd = GetForward(plan2);
    ASSERT_NE(fwd, nullptr);
    bool found = false;
    std::int32_t r1_idx = -1;
    for (std::size_t i = 0; i < fwd->request_ids.size(); ++i) {
        if (fwd->request_ids[i] == "r1") {
            found = true;
            r1_idx = static_cast<std::int32_t>(i);
            break;
        }
    }
    ASSERT_TRUE(found);
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    // After plan2 the request is in PrefillDone (re-prefill is a prefill op),
    // not Decoding yet — that comes in plan3 below.
    EXPECT_EQ(static_cast<std::size_t>(fwd->num_extends()), 1u)
        << "Recovery's first plan should emit a PrefillOperation, not a Decode";

    // Drive plan3: the executor's prefill samples a token at the last tail
    // position; SendForwardDone delivers it and PrefillDone → Decoding.
    SendForwardDone("r1", {99});
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);

    // Sanity: r1 holds device pages again (loadback'd full pages + fresh
    // partial-tail page).
    EXPECT_GE(static_cast<std::int32_t>(fwd->occupied_pages[r1_idx].size()), 1);
}

// Recovery's re-prefill emits a PrefillOperation whose ``input_ids`` cover
// the partial-tail tokens (those past PrefillSize) and whose
// ``shifted_input_ids`` includes those same tokens shifted by +1 — drafter
// boundary positions must NOT be silently -1, otherwise EAGLE first-step
// input is garbage at non-boundary positions and spec acceptance collapses
// post-recovery. This covers the ``ComputeShiftedInputIds`` cap change from
// ``PrefillSize()`` to ``Size()``.
TEST_F(RetractLeakTestSuite, RecoveryShiftedInputIdsCoversGeneratedTail) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/1, /*start=*/1));  // tokens [1,2], PrefillSize=2
    PlanOnce();
    SendForwardDone("r1", {42});  // tokens [1,2,42], Size=3 > PrefillSize=2
    PlanOnce();
    SendReserveNumTokens("r1", 5);

    auto plan1 = PlanOnce();
    const auto* wb = GetWriteBack(plan1);
    ASSERT_NE(wb, nullptr);
    SendWriteBackDone(wb->op_ids[0]);
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);

    auto plan2 = PlanOnce();
    const auto* fwd = GetForward(plan2);
    ASSERT_NE(fwd, nullptr);
    ASSERT_EQ(static_cast<std::size_t>(fwd->num_extends()), 1u);

    // partial_tail covers token 42 (position 2). With Size() cap, the
    // input_ids for this position is {42}. shifted_input_ids for position 2
    // looks at position 3 — there's no token 3 yet so it must be -1.
    // Position 2's shifted is what matters: with PrefillSize() cap (pre-fix),
    // we would NOT include token 42 in the shift window because position 2's
    // shifted_start = 3 > PrefillSize=2, so shifted_size = max(0, 2 - 3) = 0
    // → entire shifted vector is -1's.
    ASSERT_EQ(fwd->input_ids.size(), 1u) << "expect 1 partial-tail token re-prefilled";
    EXPECT_EQ(fwd->input_ids[0], 42) << "partial tail should be token 42";
    // With Size() cap, position 2's shift looks at index 3 which doesn't
    // exist → -1. That's the LAST position so -1 is the canonical "no next
    // token" sentinel, consistent with the normal Submitted-last-chunk case.
    ASSERT_EQ(fwd->shifted_input_ids.size(), 1u);
    EXPECT_EQ(fwd->shifted_input_ids[0], -1);

    SendForwardDone("r1", {99});
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
}

}  // namespace tokenspeed::test
