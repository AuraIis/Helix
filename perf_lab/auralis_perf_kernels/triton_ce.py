"""Triton forward prototype for huge-vocab linear cross entropy."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency probe
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _rowwise_linear_ce_forward_kernel(
        hidden,
        weight,
        labels,
        row_max_out,
        exp_sum_out,
        loss_out,
        D: tl.constexpr,
        V: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        label = tl.load(labels + row)
        valid = label != IGNORE_INDEX
        offs_v = tl.arange(0, BLOCK_V)
        offs_d = tl.arange(0, BLOCK_D)

        max_val = -float("inf")
        target_logit = 0.0

        for v_start in tl.range(0, V, BLOCK_V):
            vocab = v_start + offs_v
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for d_start in tl.range(0, D, BLOCK_D):
                d = d_start + offs_d
                h = tl.load(hidden + row * D + d, mask=d < D, other=0.0)
                w = tl.load(
                    weight + vocab[:, None] * D + d[None, :],
                    mask=(vocab[:, None] < V) & (d[None, :] < D),
                    other=0.0,
                )
                acc += tl.sum(w * h[None, :], axis=1)

            acc = tl.where(vocab < V, acc, -float("inf"))
            max_val = tl.maximum(max_val, tl.max(acc, axis=0))
            target_logit += tl.sum(tl.where((vocab == label) & valid, acc, 0.0), axis=0)

        exp_sum = 0.0
        for v_start in tl.range(0, V, BLOCK_V):
            vocab = v_start + offs_v
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for d_start in tl.range(0, D, BLOCK_D):
                d = d_start + offs_d
                h = tl.load(hidden + row * D + d, mask=d < D, other=0.0)
                w = tl.load(
                    weight + vocab[:, None] * D + d[None, :],
                    mask=(vocab[:, None] < V) & (d[None, :] < D),
                    other=0.0,
                )
                acc += tl.sum(w * h[None, :], axis=1)
            acc = tl.where(vocab < V, acc, -float("inf"))
            exp_sum += tl.sum(tl.exp(acc - max_val), axis=0)

        loss = max_val + tl.log(exp_sum) - target_logit
        loss = tl.where(valid, loss, 0.0)
        tl.store(row_max_out + row, max_val)
        tl.store(exp_sum_out + row, exp_sum)
        tl.store(loss_out + row, loss)

    @triton.jit
    def _block_linear_ce_forward_kernel(
        hidden,
        weight,
        labels,
        row_max_out,
        exp_sum_out,
        loss_out,
        N: tl.constexpr,
        D: tl.constexpr,
        V: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_v = tl.arange(0, BLOCK_V)
        offs_d = tl.arange(0, BLOCK_D)
        labels_m = tl.load(labels + rows, mask=rows < N, other=IGNORE_INDEX)
        valid = (rows < N) & (labels_m != IGNORE_INDEX)

        max_vals = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        target_logits = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for v_start in tl.range(0, V, BLOCK_V):
            vocab = v_start + offs_v
            acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
            for d_start in tl.range(0, D, BLOCK_D):
                d = d_start + offs_d
                h = tl.load(
                    hidden + rows[:, None] * D + d[None, :],
                    mask=(rows[:, None] < N) & (d[None, :] < D),
                    other=0.0,
                )
                w = tl.load(
                    weight + vocab[None, :] * D + d[:, None],
                    mask=(vocab[None, :] < V) & (d[:, None] < D),
                    other=0.0,
                )
                acc += tl.dot(h, w, input_precision="ieee")
            acc = tl.where((rows[:, None] < N) & (vocab[None, :] < V), acc, -float("inf"))
            max_vals = tl.maximum(max_vals, tl.max(acc, axis=1))
            target_logits += tl.sum(
                tl.where((vocab[None, :] == labels_m[:, None]) & valid[:, None], acc, 0.0),
                axis=1,
            )

        exp_sums = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for v_start in tl.range(0, V, BLOCK_V):
            vocab = v_start + offs_v
            acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
            for d_start in tl.range(0, D, BLOCK_D):
                d = d_start + offs_d
                h = tl.load(
                    hidden + rows[:, None] * D + d[None, :],
                    mask=(rows[:, None] < N) & (d[None, :] < D),
                    other=0.0,
                )
                w = tl.load(
                    weight + vocab[None, :] * D + d[:, None],
                    mask=(vocab[None, :] < V) & (d[:, None] < D),
                    other=0.0,
                )
                acc += tl.dot(h, w, input_precision="ieee")
            acc = tl.where((rows[:, None] < N) & (vocab[None, :] < V), acc, -float("inf"))
            exp_sums += tl.sum(tl.exp(acc - max_vals[:, None]), axis=1)

        losses = max_vals + tl.log(exp_sums) - target_logits
        losses = tl.where(valid, losses, 0.0)
        tl.store(row_max_out + rows, max_vals, mask=rows < N)
        tl.store(exp_sum_out + rows, exp_sums, mask=rows < N)
        tl.store(loss_out + rows, losses, mask=rows < N)

    @triton.jit
    def _parallel_block_stats_kernel(
        hidden,
        weight,
        labels,
        partial_max,
        partial_exp_sum,
        partial_target,
        N: tl.constexpr,
        D: tl.constexpr,
        V: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        NUM_V_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
        ROUND_BF16_LOGITS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_v = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        vocab = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        offs_d = tl.arange(0, BLOCK_D)
        labels_m = tl.load(labels + rows, mask=rows < N, other=IGNORE_INDEX)
        valid = (rows < N) & (labels_m != IGNORE_INDEX)

        acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
        for d_start in tl.range(0, D, BLOCK_D):
            d = d_start + offs_d
            h = tl.load(
                hidden + rows[:, None] * D + d[None, :],
                mask=(rows[:, None] < N) & (d[None, :] < D),
                other=0.0,
            )
            w = tl.load(
                weight + vocab[None, :] * D + d[:, None],
                mask=(vocab[None, :] < V) & (d[:, None] < D),
                other=0.0,
            )
            acc += tl.dot(h, w, input_precision="ieee")

        if ROUND_BF16_LOGITS:
            acc = acc.to(tl.bfloat16).to(tl.float32)
        acc = tl.where((rows[:, None] < N) & (vocab[None, :] < V), acc, -float("inf"))
        block_max = tl.max(acc, axis=1)
        safe_block_max = tl.where(valid, block_max, 0.0)
        block_exp_sum = tl.sum(tl.exp(acc - safe_block_max[:, None]), axis=1)
        block_target = tl.sum(
            tl.where((vocab[None, :] == labels_m[:, None]) & valid[:, None], acc, 0.0),
            axis=1,
        )
        block_max = tl.where(valid, block_max, -float("inf"))
        block_exp_sum = tl.where(valid, block_exp_sum, 0.0)
        block_target = tl.where(valid, block_target, 0.0)

        out = (pid_m * NUM_V_BLOCKS + pid_v) * BLOCK_M + tl.arange(0, BLOCK_M)
        tl.store(partial_max + out, block_max, mask=rows < N)
        tl.store(partial_exp_sum + out, block_exp_sum, mask=rows < N)
        tl.store(partial_target + out, block_target, mask=rows < N)

    @triton.jit
    def _parallel_reduce_loss_kernel(
        labels,
        partial_max,
        partial_exp_sum,
        partial_target,
        row_max_out,
        exp_sum_out,
        loss_out,
        N: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        NUM_V_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        blocks = tl.arange(0, BLOCK_B)
        labels_m = tl.load(labels + rows, mask=rows < N, other=IGNORE_INDEX)
        valid = (rows < N) & (labels_m != IGNORE_INDEX)
        offs = (pid_m * NUM_V_BLOCKS + blocks[None, :]) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        mask = valid[:, None] & (blocks[None, :] < NUM_V_BLOCKS)

        pmax = tl.load(partial_max + offs, mask=mask, other=-float("inf"))
        psum = tl.load(partial_exp_sum + offs, mask=mask, other=0.0)
        ptarget = tl.load(partial_target + offs, mask=mask, other=0.0)

        max_vals = tl.max(pmax, axis=1)
        safe_max = tl.where(valid, max_vals, 0.0)
        exp_sums = tl.sum(tl.exp(pmax - safe_max[:, None]) * psum, axis=1)
        target_logits = tl.sum(ptarget, axis=1)
        losses = safe_max + tl.log(exp_sums) - target_logits
        losses = tl.where(valid, losses, 0.0)
        exp_sums = tl.where(valid, exp_sums, 0.0)

        tl.store(row_max_out + rows, safe_max, mask=rows < N)
        tl.store(exp_sum_out + rows, exp_sums, mask=rows < N)
        tl.store(loss_out + rows, losses, mask=rows < N)

    @triton.jit
    def _parallel_linear_ce_backward_kernel(
        hidden,
        weight,
        labels,
        row_max,
        exp_sum,
        grad_output,
        grad_hidden,
        grad_weight,
        N: tl.constexpr,
        D: tl.constexpr,
        V: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
        ROUND_BF16_LOGITS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_v = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        vocab = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        offs_d = tl.arange(0, BLOCK_D)

        labels_m = tl.load(labels + rows, mask=rows < N, other=IGNORE_INDEX)
        valid_rows = (rows < N) & (labels_m != IGNORE_INDEX)
        row_max_m = tl.load(row_max + rows, mask=rows < N, other=0.0)
        exp_sum_m = tl.load(exp_sum + rows, mask=rows < N, other=1.0)
        exp_sum_m = tl.where(valid_rows, exp_sum_m, 1.0)
        scale = tl.load(grad_output)

        logits = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
        for d_start in tl.range(0, D, BLOCK_D):
            d = d_start + offs_d
            h = tl.load(
                hidden + rows[:, None] * D + d[None, :],
                mask=(rows[:, None] < N) & (d[None, :] < D),
                other=0.0,
            )
            w = tl.load(
                weight + vocab[None, :] * D + d[:, None],
                mask=(vocab[None, :] < V) & (d[:, None] < D),
                other=0.0,
            )
            logits += tl.dot(h, w, input_precision="ieee")

        if ROUND_BF16_LOGITS:
            logits = logits.to(tl.bfloat16).to(tl.float32)
        valid_vocab = vocab < V
        logits = tl.where(valid_rows[:, None] & valid_vocab[None, :], logits, -float("inf"))
        probs = tl.exp(logits - row_max_m[:, None]) / exp_sum_m[:, None]
        one_hot = (vocab[None, :] == labels_m[:, None]) & valid_rows[:, None] & valid_vocab[None, :]
        grad_logits = tl.where(
            valid_rows[:, None] & valid_vocab[None, :],
            (probs - tl.where(one_hot, 1.0, 0.0)) * scale,
            0.0,
        )

        for d_start in tl.range(0, D, BLOCK_D):
            d = d_start + offs_d
            h = tl.load(
                hidden + rows[:, None] * D + d[None, :],
                mask=(rows[:, None] < N) & (d[None, :] < D),
                other=0.0,
            )
            w_vd = tl.load(
                weight + vocab[:, None] * D + d[None, :],
                mask=(vocab[:, None] < V) & (d[None, :] < D),
                other=0.0,
            )
            grad_h = tl.dot(grad_logits, w_vd.to(tl.float32), input_precision="ieee")
            grad_w = tl.dot(tl.trans(grad_logits), h.to(tl.float32), input_precision="ieee")
            tl.atomic_add(
                grad_hidden + rows[:, None] * D + d[None, :],
                grad_h,
                sem="relaxed",
                mask=(rows[:, None] < N) & (d[None, :] < D),
            )
            tl.atomic_add(
                grad_weight + vocab[:, None] * D + d[None, :],
                grad_w,
                sem="relaxed",
                mask=(vocab[:, None] < V) & (d[None, :] < D),
            )

    @triton.jit
    def _parallel_linear_ce_backward_weight_kernel(
        hidden,
        weight,
        labels,
        row_max,
        exp_sum,
        grad_output,
        grad_weight,
        N: tl.constexpr,
        D: tl.constexpr,
        V: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_v = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        vocab = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        offs_d = tl.arange(0, BLOCK_D)

        labels_m = tl.load(labels + rows, mask=rows < N, other=IGNORE_INDEX)
        valid_rows = (rows < N) & (labels_m != IGNORE_INDEX)
        row_max_m = tl.load(row_max + rows, mask=rows < N, other=0.0)
        exp_sum_m = tl.load(exp_sum + rows, mask=rows < N, other=1.0)
        exp_sum_m = tl.where(valid_rows, exp_sum_m, 1.0)
        scale = tl.load(grad_output)

        logits = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
        for d_start in tl.range(0, D, BLOCK_D):
            d = d_start + offs_d
            h = tl.load(
                hidden + rows[:, None] * D + d[None, :],
                mask=(rows[:, None] < N) & (d[None, :] < D),
                other=0.0,
            )
            w = tl.load(
                weight + vocab[None, :] * D + d[:, None],
                mask=(vocab[None, :] < V) & (d[:, None] < D),
                other=0.0,
            )
            logits += tl.dot(h, w, input_precision="ieee")

        valid_vocab = vocab < V
        logits = tl.where(valid_rows[:, None] & valid_vocab[None, :], logits, -float("inf"))
        probs = tl.exp(logits - row_max_m[:, None]) / exp_sum_m[:, None]
        one_hot = (vocab[None, :] == labels_m[:, None]) & valid_rows[:, None] & valid_vocab[None, :]
        grad_logits = tl.where(
            valid_rows[:, None] & valid_vocab[None, :],
            (probs - tl.where(one_hot, 1.0, 0.0)) * scale,
            0.0,
        )

        for d_start in tl.range(0, D, BLOCK_D):
            d = d_start + offs_d
            h = tl.load(
                hidden + rows[:, None] * D + d[None, :],
                mask=(rows[:, None] < N) & (d[None, :] < D),
                other=0.0,
            )
            grad_w = tl.dot(tl.trans(grad_logits), h.to(tl.float32), input_precision="ieee")
            tl.atomic_add(
                grad_weight + vocab[:, None] * D + d[None, :],
                grad_w,
                sem="relaxed",
                mask=(vocab[:, None] < V) & (d[None, :] < D),
            )

    @triton.jit
    def _parallel_linear_ce_backward_hidden_atomic_kernel(
        hidden,
        weight,
        labels,
        row_max,
        exp_sum,
        grad_output,
        grad_hidden,
        N: tl.constexpr,
        D: tl.constexpr,
        V: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_v = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        vocab = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        offs_d = tl.arange(0, BLOCK_D)

        labels_m = tl.load(labels + rows, mask=rows < N, other=IGNORE_INDEX)
        valid_rows = (rows < N) & (labels_m != IGNORE_INDEX)
        row_max_m = tl.load(row_max + rows, mask=rows < N, other=0.0)
        exp_sum_m = tl.load(exp_sum + rows, mask=rows < N, other=1.0)
        exp_sum_m = tl.where(valid_rows, exp_sum_m, 1.0)
        scale = tl.load(grad_output)

        logits = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
        for d_start in tl.range(0, D, BLOCK_D):
            d = d_start + offs_d
            h = tl.load(
                hidden + rows[:, None] * D + d[None, :],
                mask=(rows[:, None] < N) & (d[None, :] < D),
                other=0.0,
            )
            w = tl.load(
                weight + vocab[None, :] * D + d[:, None],
                mask=(vocab[None, :] < V) & (d[:, None] < D),
                other=0.0,
            )
            logits += tl.dot(h, w, input_precision="ieee")

        valid_vocab = vocab < V
        logits = tl.where(valid_rows[:, None] & valid_vocab[None, :], logits, -float("inf"))
        probs = tl.exp(logits - row_max_m[:, None]) / exp_sum_m[:, None]
        one_hot = (vocab[None, :] == labels_m[:, None]) & valid_rows[:, None] & valid_vocab[None, :]
        grad_logits = tl.where(
            valid_rows[:, None] & valid_vocab[None, :],
            (probs - tl.where(one_hot, 1.0, 0.0)) * scale,
            0.0,
        )

        for d_start in tl.range(0, D, BLOCK_D):
            d = d_start + offs_d
            w_vd = tl.load(
                weight + vocab[:, None] * D + d[None, :],
                mask=(vocab[:, None] < V) & (d[None, :] < D),
                other=0.0,
            )
            grad_h = tl.dot(grad_logits, w_vd.to(tl.float32), input_precision="ieee")
            tl.atomic_add(
                grad_hidden + rows[:, None] * D + d[None, :],
                grad_h,
                sem="relaxed",
                mask=(rows[:, None] < N) & (d[None, :] < D),
            )

    @triton.jit
    def _parallel_linear_ce_backward_weight_no_atomic_kernel(
        hidden,
        weight,
        labels,
        row_max,
        exp_sum,
        grad_output,
        grad_weight,
        N: tl.constexpr,
        D: tl.constexpr,
        V: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_v = tl.program_id(0)
        pid_d = tl.program_id(1)
        vocab = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        d_out = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        rows_base = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_D)
        scale = tl.load(grad_output)
        grad_w = tl.zeros((BLOCK_V, BLOCK_D), dtype=tl.float32)

        for row_start in tl.range(0, N, BLOCK_M):
            rows = row_start + rows_base
            labels_m = tl.load(labels + rows, mask=rows < N, other=IGNORE_INDEX)
            valid_rows = (rows < N) & (labels_m != IGNORE_INDEX)
            row_max_m = tl.load(row_max + rows, mask=rows < N, other=0.0)
            exp_sum_m = tl.load(exp_sum + rows, mask=rows < N, other=1.0)
            exp_sum_m = tl.where(valid_rows, exp_sum_m, 1.0)

            logits = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
            for k_start in tl.range(0, D, BLOCK_D):
                k = k_start + offs_k
                h_all = tl.load(
                    hidden + rows[:, None] * D + k[None, :],
                    mask=(rows[:, None] < N) & (k[None, :] < D),
                    other=0.0,
                )
                w_all = tl.load(
                    weight + vocab[None, :] * D + k[:, None],
                    mask=(vocab[None, :] < V) & (k[:, None] < D),
                    other=0.0,
                )
                logits += tl.dot(h_all, w_all, input_precision="ieee")

            valid_vocab = vocab < V
            logits = tl.where(valid_rows[:, None] & valid_vocab[None, :], logits, -float("inf"))
            probs = tl.exp(logits - row_max_m[:, None]) / exp_sum_m[:, None]
            one_hot = (vocab[None, :] == labels_m[:, None]) & valid_rows[:, None] & valid_vocab[None, :]
            grad_logits = tl.where(
                valid_rows[:, None] & valid_vocab[None, :],
                (probs - tl.where(one_hot, 1.0, 0.0)) * scale,
                0.0,
            )
            h_out = tl.load(
                hidden + rows[:, None] * D + d_out[None, :],
                mask=(rows[:, None] < N) & (d_out[None, :] < D),
                other=0.0,
            )
            grad_w += tl.dot(tl.trans(grad_logits), h_out.to(tl.float32), input_precision="ieee")

        tl.store(
            grad_weight + vocab[:, None] * D + d_out[None, :],
            grad_w,
            mask=(vocab[:, None] < V) & (d_out[None, :] < D),
        )

    @triton.jit
    def _parallel_linear_ce_backward_hidden_no_atomic_kernel(
        hidden,
        weight,
        labels,
        row_max,
        exp_sum,
        grad_output,
        grad_hidden,
        N: tl.constexpr,
        D: tl.constexpr,
        V: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d_out = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_v = tl.arange(0, BLOCK_V)
        offs_k = tl.arange(0, BLOCK_D)

        labels_m = tl.load(labels + rows, mask=rows < N, other=IGNORE_INDEX)
        valid_rows = (rows < N) & (labels_m != IGNORE_INDEX)
        row_max_m = tl.load(row_max + rows, mask=rows < N, other=0.0)
        exp_sum_m = tl.load(exp_sum + rows, mask=rows < N, other=1.0)
        exp_sum_m = tl.where(valid_rows, exp_sum_m, 1.0)
        scale = tl.load(grad_output)
        grad_h = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for v_start in tl.range(0, V, BLOCK_V):
            vocab = v_start + offs_v
            logits = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
            for k_start in tl.range(0, D, BLOCK_D):
                k = k_start + offs_k
                h_all = tl.load(
                    hidden + rows[:, None] * D + k[None, :],
                    mask=(rows[:, None] < N) & (k[None, :] < D),
                    other=0.0,
                )
                w_all = tl.load(
                    weight + vocab[None, :] * D + k[:, None],
                    mask=(vocab[None, :] < V) & (k[:, None] < D),
                    other=0.0,
                )
                logits += tl.dot(h_all, w_all, input_precision="ieee")

            valid_vocab = vocab < V
            logits = tl.where(valid_rows[:, None] & valid_vocab[None, :], logits, -float("inf"))
            probs = tl.exp(logits - row_max_m[:, None]) / exp_sum_m[:, None]
            one_hot = (vocab[None, :] == labels_m[:, None]) & valid_rows[:, None] & valid_vocab[None, :]
            grad_logits = tl.where(
                valid_rows[:, None] & valid_vocab[None, :],
                (probs - tl.where(one_hot, 1.0, 0.0)) * scale,
                0.0,
            )
            w_out = tl.load(
                weight + vocab[:, None] * D + d_out[None, :],
                mask=(vocab[:, None] < V) & (d_out[None, :] < D),
                other=0.0,
            )
            grad_h += tl.dot(grad_logits, w_out.to(tl.float32), input_precision="ieee")

        tl.store(
            grad_hidden + rows[:, None] * D + d_out[None, :],
            grad_h,
            mask=(rows[:, None] < N) & (d_out[None, :] < D),
        )

    @triton.jit
    def _parallel_linear_ce_backward_weight_partial_kernel(
        hidden,
        weight,
        labels,
        row_max,
        exp_sum,
        grad_output,
        partial_grad_weight,
        N: tl.constexpr,
        D: tl.constexpr,
        V: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        ROW_GROUP_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_v = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_g = tl.program_id(2)
        vocab = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        d_out = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        rows_base = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_D)
        scale = tl.load(grad_output)
        grad_w = tl.zeros((BLOCK_V, BLOCK_D), dtype=tl.float32)
        group_row_start = pid_g * ROW_GROUP_BLOCKS * BLOCK_M

        for group_i in tl.range(0, ROW_GROUP_BLOCKS):
            rows = group_row_start + group_i * BLOCK_M + rows_base
            labels_m = tl.load(labels + rows, mask=rows < N, other=IGNORE_INDEX)
            valid_rows = (rows < N) & (labels_m != IGNORE_INDEX)
            row_max_m = tl.load(row_max + rows, mask=rows < N, other=0.0)
            exp_sum_m = tl.load(exp_sum + rows, mask=rows < N, other=1.0)
            exp_sum_m = tl.where(valid_rows, exp_sum_m, 1.0)

            logits = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
            for k_start in tl.range(0, D, BLOCK_D):
                k = k_start + offs_k
                h_all = tl.load(
                    hidden + rows[:, None] * D + k[None, :],
                    mask=(rows[:, None] < N) & (k[None, :] < D),
                    other=0.0,
                )
                w_all = tl.load(
                    weight + vocab[None, :] * D + k[:, None],
                    mask=(vocab[None, :] < V) & (k[:, None] < D),
                    other=0.0,
                )
                logits += tl.dot(h_all, w_all, input_precision="ieee")

            valid_vocab = vocab < V
            logits = tl.where(valid_rows[:, None] & valid_vocab[None, :], logits, -float("inf"))
            probs = tl.exp(logits - row_max_m[:, None]) / exp_sum_m[:, None]
            one_hot = (vocab[None, :] == labels_m[:, None]) & valid_rows[:, None] & valid_vocab[None, :]
            grad_logits = tl.where(
                valid_rows[:, None] & valid_vocab[None, :],
                (probs - tl.where(one_hot, 1.0, 0.0)) * scale,
                0.0,
            )
            h_out = tl.load(
                hidden + rows[:, None] * D + d_out[None, :],
                mask=(rows[:, None] < N) & (d_out[None, :] < D),
                other=0.0,
            )
            grad_w += tl.dot(tl.trans(grad_logits), h_out.to(tl.float32), input_precision="ieee")

        tl.store(
            partial_grad_weight + pid_g * V * D + vocab[:, None] * D + d_out[None, :],
            grad_w,
            mask=(vocab[:, None] < V) & (d_out[None, :] < D),
        )

    @triton.jit
    def _reduce_grad_weight_partials_kernel(
        partial_grad_weight,
        grad_weight,
        D: tl.constexpr,
        V: tl.constexpr,
        NUM_ROW_GROUPS: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_v = tl.program_id(0)
        pid_d = tl.program_id(1)
        vocab = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        grad_w = tl.zeros((BLOCK_V, BLOCK_D), dtype=tl.float32)

        for group_i in tl.range(0, NUM_ROW_GROUPS):
            grad_w += tl.load(
                partial_grad_weight + group_i * V * D + vocab[:, None] * D + d[None, :],
                mask=(vocab[:, None] < V) & (d[None, :] < D),
                other=0.0,
            )

        tl.store(
            grad_weight + vocab[:, None] * D + d[None, :],
            grad_w,
            mask=(vocab[:, None] < V) & (d[None, :] < D),
        )


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def triton_linear_ce_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    block_v: int = 64,
    block_d: int = 64,
    block_m: int = 8,
    num_warps: int = 4,
    num_stages: int = 3,
    mode: str = "block",
    round_bf16_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(losses, row_max, exp_sum)` for CE over `hidden @ weight.T`.

    This is a forward-only research prototype. It intentionally computes one
    token row per Triton program and loops across vocab chunks. That keeps the
    code small and testable; it is not expected to beat cuBLAS on throughput yet.
    """

    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton is not available")
    if not hidden.is_cuda or not weight.is_cuda or not labels.is_cuda:
        raise ValueError("hidden, weight and labels must be CUDA tensors")
    if hidden.dim() != 2 or weight.dim() != 2:
        raise ValueError("hidden and weight must be 2D")
    if hidden.size(1) != weight.size(1):
        raise ValueError("hidden dim must match weight dim")
    labels = labels.reshape(-1).contiguous()
    if labels.numel() != hidden.size(0):
        raise ValueError("labels length must match hidden rows")

    hidden = hidden.contiguous()
    weight = weight.contiguous()
    tokens = hidden.size(0)
    d_model = hidden.size(1)
    vocab_size = weight.size(0)
    row_max = torch.empty((tokens,), device=hidden.device, dtype=torch.float32)
    exp_sum = torch.empty((tokens,), device=hidden.device, dtype=torch.float32)
    losses = torch.empty((tokens,), device=hidden.device, dtype=torch.float32)
    if mode == "row":
        _rowwise_linear_ce_forward_kernel[(tokens,)](
            hidden,
            weight,
            labels,
            row_max,
            exp_sum,
            losses,
            D=d_model,
            V=vocab_size,
            IGNORE_INDEX=int(ignore_index),
            BLOCK_V=int(block_v),
            BLOCK_D=int(block_d),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        )
    elif mode == "block":
        grid = (triton.cdiv(tokens, int(block_m)),)
        _block_linear_ce_forward_kernel[grid](
            hidden,
            weight,
            labels,
            row_max,
            exp_sum,
            losses,
            N=tokens,
            D=d_model,
            V=vocab_size,
            IGNORE_INDEX=int(ignore_index),
            BLOCK_M=int(block_m),
            BLOCK_V=int(block_v),
            BLOCK_D=int(block_d),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        )
    elif mode == "parallel":
        num_m_blocks = triton.cdiv(tokens, int(block_m))
        num_v_blocks = triton.cdiv(vocab_size, int(block_v))
        partial_shape = (num_m_blocks * num_v_blocks * int(block_m),)
        partial_max = torch.empty(partial_shape, device=hidden.device, dtype=torch.float32)
        partial_exp_sum = torch.empty(partial_shape, device=hidden.device, dtype=torch.float32)
        partial_target = torch.empty(partial_shape, device=hidden.device, dtype=torch.float32)
        _parallel_block_stats_kernel[(num_m_blocks, num_v_blocks)](
            hidden,
            weight,
            labels,
            partial_max,
            partial_exp_sum,
            partial_target,
            N=tokens,
            D=d_model,
            V=vocab_size,
            IGNORE_INDEX=int(ignore_index),
            NUM_V_BLOCKS=num_v_blocks,
            BLOCK_M=int(block_m),
            BLOCK_V=int(block_v),
            BLOCK_D=int(block_d),
            ROUND_BF16_LOGITS=bool(round_bf16_logits),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        )
        _parallel_reduce_loss_kernel[(num_m_blocks,)](
            labels,
            partial_max,
            partial_exp_sum,
            partial_target,
            row_max,
            exp_sum,
            losses,
            N=tokens,
            IGNORE_INDEX=int(ignore_index),
            NUM_V_BLOCKS=num_v_blocks,
            BLOCK_M=int(block_m),
            BLOCK_B=_next_power_of_2(num_v_blocks),
            num_warps=int(num_warps),
            num_stages=1,
        )
    else:
        raise ValueError(f"unknown Triton CE mode: {mode}")
    return losses, row_max, exp_sum


def triton_linear_ce_backward(
    grad_output: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    row_max: torch.Tensor,
    exp_sum: torch.Tensor,
    valid_count: torch.Tensor,
    *,
    ignore_index: int = -100,
    block_m: int = 16,
    block_v: int = 32,
    block_d: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
    round_bf16_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gradients for the Triton linear-CE prototype.

    This is the first fused backward research path. It recomputes logits per
    `(row block, vocab block)` and atomically accumulates `grad_hidden` and
    `grad_weight`, avoiding full-logits materialization. The current kernel is
    designed for correctness and memory validation before speed tuning.
    """

    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton is not available")
    if not hidden.is_cuda or not weight.is_cuda or not labels.is_cuda:
        raise ValueError("hidden, weight and labels must be CUDA tensors")
    if hidden.dim() != 2 or weight.dim() != 2:
        raise ValueError("hidden and weight must be 2D")
    if hidden.size(1) != weight.size(1):
        raise ValueError("hidden dim must match weight dim")
    labels = labels.reshape(-1).contiguous()
    if labels.numel() != hidden.size(0):
        raise ValueError("labels length must match hidden rows")

    block_m = max(int(block_m), 16)
    block_v = max(int(block_v), 16)
    block_d = max(int(block_d), 16)
    hidden = hidden.contiguous()
    weight = weight.contiguous()
    tokens = hidden.size(0)
    d_model = hidden.size(1)
    vocab_size = weight.size(0)
    scale = (grad_output.float() / valid_count.to(torch.float32)).reshape(())
    grad_hidden = torch.zeros_like(hidden, dtype=torch.float32)
    grad_weight = torch.zeros_like(weight, dtype=torch.float32)
    grid = (triton.cdiv(tokens, block_m), triton.cdiv(vocab_size, block_v))
    _parallel_linear_ce_backward_kernel[grid](
        hidden,
        weight,
        labels,
        row_max.contiguous(),
        exp_sum.contiguous(),
        scale,
        grad_hidden,
        grad_weight,
        N=tokens,
        D=d_model,
        V=vocab_size,
        IGNORE_INDEX=int(ignore_index),
        BLOCK_M=block_m,
        BLOCK_V=block_v,
        BLOCK_D=block_d,
        ROUND_BF16_LOGITS=bool(round_bf16_logits),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return grad_hidden.to(dtype=hidden.dtype), grad_weight.to(dtype=weight.dtype)


def triton_linear_ce_backward_atomic_lowmem(
    grad_output: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    row_max: torch.Tensor,
    exp_sum: torch.Tensor,
    valid_count: torch.Tensor,
    *,
    ignore_index: int = -100,
    block_m: int = 16,
    block_v: int = 32,
    block_d: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
    round_bf16_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Atomic backward with gradient buffers in input dtype.

    The default atomic path accumulates `grad_hidden` and `grad_weight` into
    fp32 buffers before casting back to the parameter dtype. That is safer for
    precision but costly for huge vocabularies. This low-memory variant uses
    the same fused kernel with bf16/fp16/fp32 output buffers matching the inputs.
    """

    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton is not available")
    if not hidden.is_cuda or not weight.is_cuda or not labels.is_cuda:
        raise ValueError("hidden, weight and labels must be CUDA tensors")
    if hidden.dim() != 2 or weight.dim() != 2:
        raise ValueError("hidden and weight must be 2D")
    if hidden.size(1) != weight.size(1):
        raise ValueError("hidden dim must match weight dim")
    labels = labels.reshape(-1).contiguous()
    if labels.numel() != hidden.size(0):
        raise ValueError("labels length must match hidden rows")

    block_m = max(int(block_m), 16)
    block_v = max(int(block_v), 16)
    block_d = max(int(block_d), 16)
    hidden = hidden.contiguous()
    weight = weight.contiguous()
    tokens = hidden.size(0)
    d_model = hidden.size(1)
    vocab_size = weight.size(0)
    scale = (grad_output.float() / valid_count.to(torch.float32)).reshape(())
    grad_hidden = torch.zeros_like(hidden)
    grad_weight = torch.zeros_like(weight)
    grid = (triton.cdiv(tokens, block_m), triton.cdiv(vocab_size, block_v))
    _parallel_linear_ce_backward_kernel[grid](
        hidden,
        weight,
        labels,
        row_max.contiguous(),
        exp_sum.contiguous(),
        scale,
        grad_hidden,
        grad_weight,
        N=tokens,
        D=d_model,
        V=vocab_size,
        IGNORE_INDEX=int(ignore_index),
        BLOCK_M=block_m,
        BLOCK_V=block_v,
        BLOCK_D=block_d,
        ROUND_BF16_LOGITS=bool(round_bf16_logits),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return grad_hidden, grad_weight


def triton_linear_ce_backward_atomic_mixed(
    grad_output: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    row_max: torch.Tensor,
    exp_sum: torch.Tensor,
    valid_count: torch.Tensor,
    *,
    ignore_index: int = -100,
    block_m: int = 16,
    block_v: int = 32,
    block_d: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
    round_bf16_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Atomic backward with fp32 `grad_hidden` and low-memory `grad_weight`.

    `atomic_lowmem` saves memory by accumulating both gradients directly into
    bf16/fp16 buffers. Loss-drift tests showed most harmful drift appears in the
    upstream hidden gradient. This mixed path keeps that accumulation in fp32
    and only applies the low-memory shortcut to the huge vocab weight gradient.
    """

    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton is not available")
    if not hidden.is_cuda or not weight.is_cuda or not labels.is_cuda:
        raise ValueError("hidden, weight and labels must be CUDA tensors")
    if hidden.dim() != 2 or weight.dim() != 2:
        raise ValueError("hidden and weight must be 2D")
    if hidden.size(1) != weight.size(1):
        raise ValueError("hidden dim must match weight dim")
    labels = labels.reshape(-1).contiguous()
    if labels.numel() != hidden.size(0):
        raise ValueError("labels length must match hidden rows")

    block_m = max(int(block_m), 16)
    block_v = max(int(block_v), 16)
    block_d = max(int(block_d), 16)
    hidden = hidden.contiguous()
    weight = weight.contiguous()
    tokens = hidden.size(0)
    d_model = hidden.size(1)
    vocab_size = weight.size(0)
    scale = (grad_output.float() / valid_count.to(torch.float32)).reshape(())
    grad_hidden = torch.zeros_like(hidden, dtype=torch.float32)
    grad_weight = torch.zeros_like(weight)
    grid = (triton.cdiv(tokens, block_m), triton.cdiv(vocab_size, block_v))
    _parallel_linear_ce_backward_kernel[grid](
        hidden,
        weight,
        labels,
        row_max.contiguous(),
        exp_sum.contiguous(),
        scale,
        grad_hidden,
        grad_weight,
        N=tokens,
        D=d_model,
        V=vocab_size,
        IGNORE_INDEX=int(ignore_index),
        BLOCK_M=block_m,
        BLOCK_V=block_v,
        BLOCK_D=block_d,
        ROUND_BF16_LOGITS=bool(round_bf16_logits),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return grad_hidden.to(dtype=hidden.dtype), grad_weight


def triton_linear_ce_backward_split_hidden(
    grad_output: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    row_max: torch.Tensor,
    exp_sum: torch.Tensor,
    valid_count: torch.Tensor,
    *,
    ignore_index: int = -100,
    block_m: int = 16,
    block_v: int = 32,
    block_d: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward variant with no atomics for `grad_hidden`.

    `grad_hidden` is computed by row/dim tiles that own their output region and
    loop over vocab blocks. `grad_weight` still uses the original row/vocab
    tiling with atomics. This intentionally trades extra recomputation for fewer
    hidden-gradient atomics so the lab can measure the break-even point.
    """

    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton is not available")
    if not hidden.is_cuda or not weight.is_cuda or not labels.is_cuda:
        raise ValueError("hidden, weight and labels must be CUDA tensors")
    if hidden.dim() != 2 or weight.dim() != 2:
        raise ValueError("hidden and weight must be 2D")
    if hidden.size(1) != weight.size(1):
        raise ValueError("hidden dim must match weight dim")
    labels = labels.reshape(-1).contiguous()
    if labels.numel() != hidden.size(0):
        raise ValueError("labels length must match hidden rows")

    block_m = max(int(block_m), 16)
    block_v = max(int(block_v), 16)
    block_d = max(int(block_d), 16)
    hidden = hidden.contiguous()
    weight = weight.contiguous()
    tokens = hidden.size(0)
    d_model = hidden.size(1)
    vocab_size = weight.size(0)
    scale = (grad_output.float() / valid_count.to(torch.float32)).reshape(())
    grad_hidden = torch.empty_like(hidden, dtype=torch.float32)
    grad_weight = torch.zeros_like(weight, dtype=torch.float32)

    _parallel_linear_ce_backward_hidden_no_atomic_kernel[
        (triton.cdiv(tokens, block_m), triton.cdiv(d_model, block_d))
    ](
        hidden,
        weight,
        labels,
        row_max.contiguous(),
        exp_sum.contiguous(),
        scale,
        grad_hidden,
        N=tokens,
        D=d_model,
        V=vocab_size,
        IGNORE_INDEX=int(ignore_index),
        BLOCK_M=block_m,
        BLOCK_V=block_v,
        BLOCK_D=block_d,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    _parallel_linear_ce_backward_weight_kernel[
        (triton.cdiv(tokens, block_m), triton.cdiv(vocab_size, block_v))
    ](
        hidden,
        weight,
        labels,
        row_max.contiguous(),
        exp_sum.contiguous(),
        scale,
        grad_weight,
        N=tokens,
        D=d_model,
        V=vocab_size,
        IGNORE_INDEX=int(ignore_index),
        BLOCK_M=block_m,
        BLOCK_V=block_v,
        BLOCK_D=block_d,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return grad_hidden.to(dtype=hidden.dtype), grad_weight.to(dtype=weight.dtype)


def triton_linear_ce_backward_split_weight(
    grad_output: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    row_max: torch.Tensor,
    exp_sum: torch.Tensor,
    valid_count: torch.Tensor,
    *,
    ignore_index: int = -100,
    block_m: int = 16,
    block_v: int = 32,
    block_d: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward variant with no atomics for `grad_weight`.

    `grad_weight` is computed by vocab/dim tiles that own their output region
    and loop over all row blocks. `grad_hidden` keeps the original row/vocab
    tiling with atomics. This isolates whether weight-gradient atomics are worth
    trading for extra row-loop recomputation.
    """

    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton is not available")
    if not hidden.is_cuda or not weight.is_cuda or not labels.is_cuda:
        raise ValueError("hidden, weight and labels must be CUDA tensors")
    if hidden.dim() != 2 or weight.dim() != 2:
        raise ValueError("hidden and weight must be 2D")
    if hidden.size(1) != weight.size(1):
        raise ValueError("hidden dim must match weight dim")
    labels = labels.reshape(-1).contiguous()
    if labels.numel() != hidden.size(0):
        raise ValueError("labels length must match hidden rows")

    block_m = max(int(block_m), 16)
    block_v = max(int(block_v), 16)
    block_d = max(int(block_d), 16)
    hidden = hidden.contiguous()
    weight = weight.contiguous()
    tokens = hidden.size(0)
    d_model = hidden.size(1)
    vocab_size = weight.size(0)
    scale = (grad_output.float() / valid_count.to(torch.float32)).reshape(())
    grad_hidden = torch.zeros_like(hidden, dtype=torch.float32)
    grad_weight = torch.empty_like(weight, dtype=torch.float32)

    _parallel_linear_ce_backward_hidden_atomic_kernel[
        (triton.cdiv(tokens, block_m), triton.cdiv(vocab_size, block_v))
    ](
        hidden,
        weight,
        labels,
        row_max.contiguous(),
        exp_sum.contiguous(),
        scale,
        grad_hidden,
        N=tokens,
        D=d_model,
        V=vocab_size,
        IGNORE_INDEX=int(ignore_index),
        BLOCK_M=block_m,
        BLOCK_V=block_v,
        BLOCK_D=block_d,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    _parallel_linear_ce_backward_weight_no_atomic_kernel[
        (triton.cdiv(vocab_size, block_v), triton.cdiv(d_model, block_d))
    ](
        hidden,
        weight,
        labels,
        row_max.contiguous(),
        exp_sum.contiguous(),
        scale,
        grad_weight,
        N=tokens,
        D=d_model,
        V=vocab_size,
        IGNORE_INDEX=int(ignore_index),
        BLOCK_M=block_m,
        BLOCK_V=block_v,
        BLOCK_D=block_d,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return grad_hidden.to(dtype=hidden.dtype), grad_weight.to(dtype=weight.dtype)


def triton_linear_ce_backward_partial_weight(
    grad_output: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    row_max: torch.Tensor,
    exp_sum: torch.Tensor,
    valid_count: torch.Tensor,
    *,
    ignore_index: int = -100,
    block_m: int = 16,
    block_v: int = 32,
    block_d: int = 64,
    row_group_blocks: int = 8,
    max_partial_gb: float = 8.0,
    num_warps: int = 4,
    num_stages: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward variant with bounded partial reductions for `grad_weight`.

    This keeps the original atomic `grad_hidden` path, but computes
    `grad_weight` into `(row_group, vocab, dim)` partial buffers and reduces
    them in a second kernel. It is a lab-only midpoint between fully atomic
    weight accumulation and the slower all-rows `split_weight` recomputation.
    """

    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton is not available")
    if not hidden.is_cuda or not weight.is_cuda or not labels.is_cuda:
        raise ValueError("hidden, weight and labels must be CUDA tensors")
    if hidden.dim() != 2 or weight.dim() != 2:
        raise ValueError("hidden and weight must be 2D")
    if hidden.size(1) != weight.size(1):
        raise ValueError("hidden dim must match weight dim")
    labels = labels.reshape(-1).contiguous()
    if labels.numel() != hidden.size(0):
        raise ValueError("labels length must match hidden rows")

    block_m = max(int(block_m), 16)
    block_v = max(int(block_v), 16)
    block_d = max(int(block_d), 16)
    row_group_blocks = max(int(row_group_blocks), 1)
    hidden = hidden.contiguous()
    weight = weight.contiguous()
    tokens = hidden.size(0)
    d_model = hidden.size(1)
    vocab_size = weight.size(0)
    scale = (grad_output.float() / valid_count.to(torch.float32)).reshape(())
    grad_hidden = torch.zeros_like(hidden, dtype=torch.float32)
    grad_weight = torch.empty_like(weight, dtype=torch.float32)

    num_row_blocks = triton.cdiv(tokens, block_m)
    num_row_groups = triton.cdiv(num_row_blocks, row_group_blocks)
    partial_gb = num_row_groups * vocab_size * d_model * 4 / 1e9
    if partial_gb > float(max_partial_gb):
        raise RuntimeError(
            "partial_weight would allocate "
            f"{partial_gb:.2f}GB of partial grad_weight, above {max_partial_gb:.2f}GB; "
            "raise row_group_blocks or max_partial_gb for this experiment"
        )

    _parallel_linear_ce_backward_hidden_atomic_kernel[
        (triton.cdiv(tokens, block_m), triton.cdiv(vocab_size, block_v))
    ](
        hidden,
        weight,
        labels,
        row_max.contiguous(),
        exp_sum.contiguous(),
        scale,
        grad_hidden,
        N=tokens,
        D=d_model,
        V=vocab_size,
        IGNORE_INDEX=int(ignore_index),
        BLOCK_M=block_m,
        BLOCK_V=block_v,
        BLOCK_D=block_d,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )

    partial_grad_weight = torch.empty(
        (num_row_groups, vocab_size, d_model),
        device=weight.device,
        dtype=torch.float32,
    )
    _parallel_linear_ce_backward_weight_partial_kernel[
        (triton.cdiv(vocab_size, block_v), triton.cdiv(d_model, block_d), num_row_groups)
    ](
        hidden,
        weight,
        labels,
        row_max.contiguous(),
        exp_sum.contiguous(),
        scale,
        partial_grad_weight,
        N=tokens,
        D=d_model,
        V=vocab_size,
        IGNORE_INDEX=int(ignore_index),
        ROW_GROUP_BLOCKS=row_group_blocks,
        BLOCK_M=block_m,
        BLOCK_V=block_v,
        BLOCK_D=block_d,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    _reduce_grad_weight_partials_kernel[
        (triton.cdiv(vocab_size, block_v), triton.cdiv(d_model, block_d))
    ](
        partial_grad_weight,
        grad_weight,
        D=d_model,
        V=vocab_size,
        NUM_ROW_GROUPS=num_row_groups,
        BLOCK_V=block_v,
        BLOCK_D=block_d,
        num_warps=int(num_warps),
        num_stages=1,
    )
    return grad_hidden.to(dtype=hidden.dtype), grad_weight.to(dtype=weight.dtype)


__all__ = [
    "triton_linear_ce_backward",
    "triton_linear_ce_backward_atomic_lowmem",
    "triton_linear_ce_backward_atomic_mixed",
    "triton_linear_ce_backward_partial_weight",
    "triton_linear_ce_backward_split_hidden",
    "triton_linear_ce_backward_split_weight",
    "triton_linear_ce_forward",
]
