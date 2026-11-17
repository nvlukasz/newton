# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The scene solve's row templates and row-family builders, in the sparse mass factor's coordinates.

Two native templates carry Steel's literal text with placeholders: the contact response
(:data:`_RESPONSE_SNIPPET`), which Steel's specialised kernel
(:func:`~newton._src.solvers.steel.sparse_contact._get_sparse_contact_response_kernel`) and the scene's group-table
kernel (:func:`_get_scene_contact_response_kernel`) share, and the internal-row builders' body
(:data:`_JOINT_DOF_ROWS_TEMPLATE`), whose joint-limit expansion at one mask word is the source of Steel's
:func:`~newton._src.solvers.steel.sparse_contact.build_sparse_joint_limit_rows`. Steel's module keeps its literal
text; the scene's templates expand to it character for character at one mask word, and support masks of more words
fill the mask placeholders with the helpers of :data:`_MASK_HELPERS_CUDA`. The row families' builders (drives,
mimic, loop closures, joint limits, joint friction, joint and free-body velocity limits) are the template's other
expansions.
"""

import inspect
import re
import textwrap
from functools import cache

import warp as wp

from ...sim import BodyFlags
from .friction import contact_tangent_basis
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONNECT,
    PGS_CONSTRAINT_TYPE_JOINT_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_MIMIC,
    ROW_SLOTS_ATOMIC,
    ROW_SLOTS_COUNT,
    WARP_DETERMINISM_OFF,
    _anchor_to_world,
    _connect_rotation_rows,
    _is_identity_rotation,
    _walk_has_free_dof,
    populate_connect_J_for_size,
    reserve_world_rows,
)
from .sparse_contact import build_sparse_joint_limit_rows

wp.set_module_options(WARP_DETERMINISM_OFF)


# Support masks of more than 64 coordinates: MASK_WORDS uint64 words, word w holding coordinates
# 64 w to 64 w + 63, as a local array of words. mask_test(mask, bit) is whether coordinate bit is in
# the mask, mask_rank(mask, bit) the number of its coordinates below bit (mask_rank(mask,
# 64 * MASK_WORDS) is its popcount), and mask_next(pending) clears pending's lowest coordinate and
# returns it, or -1 once pending is empty. No shift reaches 64. A kernel declares MASK_WORDS and
# the helpers it calls (:func:`mask_helpers_snippet`); _MASK_HELPERS_CUDA is the whole block.
_MASK_HELPER_PARTS = {
    "MASK_WORDS": """    constexpr int MASK_WORDS = $MASK_WORDS;
""",
    "mask_test": """    const auto mask_test = [](const unsigned long long* mask, int bit) {
        return ((mask[bit >> 6] >> (bit & 63)) & 1ull) != 0ull;
    };
""",
    "mask_rank": """    const auto mask_rank = [](const unsigned long long* mask, int bit) {
        int count = 0;
        for (int word = 0; word < $MASK_WORDS && 64 * word < bit; ++word) {
            unsigned long long value = mask[word];
            if (bit - 64 * word < 64) value &= (1ull << (bit - 64 * word)) - 1ull;
#if defined(__CUDA_ARCH__)
            count += __popcll(value);
#else
            while (value != 0) { value &= value - 1; ++count; }
#endif
        }
        return count;
    };
""",
    "mask_next": """    const auto mask_next = [](unsigned long long* pending) {
        for (int word = 0; word < $MASK_WORDS; ++word) {
            unsigned long long value = pending[word];
            if (value == 0) continue;
            pending[word] = value & (value - 1);
#if defined(__CUDA_ARCH__)
            return 64 * word + __ffsll(value) - 1;
#else
            int bit = 0;
            while ((value & 1ull) == 0) { value >>= 1; ++bit; }
            return 64 * word + bit;
#endif
        }
        return -1;
    };
""",
}
_MASK_HELPERS_CUDA = "".join(_MASK_HELPER_PARTS.values())

_MASK_HELPERS = ("mask_test", "mask_rank", "mask_next")


def _mask_helpers_template(helpers: tuple[str, ...] = _MASK_HELPERS) -> str:
    """``MASK_WORDS`` and the ``helpers`` of :data:`_MASK_HELPERS_CUDA`, with ``$MASK_WORDS`` still to fill."""
    unknown = sorted(set(helpers) - set(_MASK_HELPERS))
    if unknown:
        raise ValueError(f"Unknown mask helpers {unknown}; the helpers are {list(_MASK_HELPERS)}")
    names = ("MASK_WORDS", *(name for name in _MASK_HELPERS if name in helpers))
    return "".join(_MASK_HELPER_PARTS[name] for name in names)


def mask_helpers_snippet(mask_words: int, helpers: tuple[str, ...] = _MASK_HELPERS) -> str:
    """``MASK_WORDS`` and the ``helpers`` of :data:`_MASK_HELPERS_CUDA` for masks of ``mask_words`` words.

    By default every helper: :data:`_MASK_HELPERS_CUDA` itself.
    """
    words = int(mask_words)
    if words < 1:
        raise ValueError(f"Support masks need at least one word, got mask_words={mask_words}")
    return _mask_helpers_template(helpers).replace("$MASK_WORDS", str(words))


def _expand_masks(snippet: str, sites: dict[str, tuple[str, str]], mask_words: int) -> str:
    """Fill the mask placeholders of ``snippet`` for masks of ``mask_words`` words.

    ``sites`` maps each placeholder to its one-word text, today's single ``uint64`` expression,
    and its multi-word text, which calls the helpers of :data:`_MASK_HELPERS_CUDA`: with one word
    the snippet is the source from before multi-word masks, character for character.
    """
    words = int(mask_words)
    if words < 1:
        raise ValueError(f"Support masks need at least one word, got mask_words={mask_words}")
    # Longer placeholders first, so that none is replaced inside another.
    for placeholder in sorted(sites, key=len, reverse=True):
        if placeholder not in snippet:
            raise AssertionError(f"mask placeholder {placeholder} is missing from the snippet")
        one_word, multi_word = sites[placeholder]
        snippet = snippet.replace(placeholder, one_word if words == 1 else multi_word)
    return snippet.replace("$MASK_WORDS", str(words))


# The contact response's native body, shared by the specialised kernel (one factor plan) and the
# group-table kernel. $N bounds the factor endpoint's coordinates; the factor branch reads its plan
# and inverse factor through $FACTOR_GROUP (lines that load the endpoint's group, or none),
# $PERMUTATION, $ROW_OFFSETS, $FACTOR_BASE and $INVERSE_FACTOR. Every other line, and so the
# arithmetic and its order, is common to the two.
_RESPONSE_SNIPPET = """
#if defined(__CUDA_ARCH__)
    const int lane = tid & ($LANES - 1);
    const int stride = $LANES;
    const int group = threadIdx.x / $LANES;
    const unsigned MASK = (0xffffffffu >> (32 - $LANES))
        << ((threadIdx.x & 31) & ~($LANES - 1));
    __shared__ float storage[$STORAGE];
    float* jacobian = storage + group * $SCRATCH;
    auto angular_a_shared = angular_a;
    auto angular_b_shared = angular_b;
    auto directions_shared = directions;
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        if (row >= row_count) continue;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            angular_a_shared.data[row][k] = __shfl_sync(MASK, angular_a.data[row][k], 0, $LANES);
            angular_b_shared.data[row][k] = __shfl_sync(MASK, angular_b.data[row][k], 0, $LANES);
            directions_shared.data[row][k] = __shfl_sync(MASK, directions.data[row][k], 0, $LANES);
        }
    }
#else
    if ((tid & ($LANES - 1)) != 0) return;
    const int lane = 0;
    const int stride = 1;
    float jacobian[$SCRATCH];
    const auto angular_a_shared = angular_a;
    const auto angular_b_shared = angular_b;
    const auto directions_shared = directions;
#endif
    float* free_factor = jacobian + $JACOBIAN;
$MASK_HELPERS    const auto jacobian_value = [&](const auto& motion, const auto& angular_wrenches, int row) {
        const auto linear = wp::vec3(motion.c[0], motion.c[1], motion.c[2]);
        const auto angular = wp::vec3(motion.c[3], motion.c[4], motion.c[5]);
        const auto direction = wp::vec3(directions_shared.data[row][0],
            directions_shared.data[row][1], directions_shared.data[row][2]);
        const auto angular_wrench = wp::vec3(angular_wrenches.data[row][0],
            angular_wrenches.data[row][1], angular_wrenches.data[row][2]);
        return wp::dot(direction, linear) + wp::dot(angular_wrench, angular);
    };
$MASK_LOADS    const int row_offset = world * row_dof.shape[1] + slot;
    const int support_capacity = row_dof.shape[2];
    int count = 0;
    auto norm = wp::vec3(0.0f);
    auto incident = wp::vec3(0.0f);
    // Physical blocks remain a short prefix; ordinary free/free contacts use
    // native MF storage, so production rows have at most one such block.
    int first_endpoint = 0;
    if (art_b >= 0 && is_free_rigid.data[art_b] != 0
        && (art_a < 0 || is_free_rigid.data[art_a] == 0)) first_endpoint = 1;
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
        const int side = (first_endpoint + endpoint) % 2;
        const int art = side == 0 ? art_a : art_b;
        if (art < 0 || (side == 1 && art == art_a)
            || articulation_response_dof_count.data[art] <= 0) continue;
        const int group = art_group_idx.data[art];
        const int start = articulation_dof_start.data[art];
        const int offset = articulation_world_dof_offset.data[art];
        if (is_free_rigid.data[art] != 0) {
            for (int node = lane; node < 6; node += stride) {
                const auto motion = joint_S_s.data[start + node];
                const float velocity = v_hat.data[start + node];
#pragma unroll
                for (int row = 0; row < 3; ++row) {
                    if (row >= row_count) continue;
                    float value = 0.0f;
                    if (art == art_a) value += jacobian_value(motion, angular_a_shared, row);
                    if (art == art_b) value -= jacobian_value(motion, angular_b_shared, row);
                    jacobian[row * $PLANE + node] = value;
                    incident.c[row] += value * velocity;
                }
            }
#if defined(__CUDA_ARCH__)
            __syncwarp(MASK);
#endif
            const int factor_base = group * 36;
            for (int node = lane; node < 6; node += stride) {
                auto value = wp::vec3(0.0f);
                for (int column = 0; column <= node; ++column) {
                    const float factor = free_inverse_factor.data[factor_base + node * 6 + column];
#pragma unroll
                    for (int row = 0; row < 3; ++row) {
                        if (row < row_count) value.c[row] += factor * jacobian[row * $PLANE + column];
                    }
                }
#pragma unroll
                for (int row = 0; row < 3; ++row) {
                    if (row < row_count) free_factor[row * $PLANE + node] = value.c[row];
                }
            }
#if defined(__CUDA_ARCH__)
            __syncwarp(MASK);
#endif
            for (int node = lane; node < 6; node += stride) {
                auto response = wp::vec3(0.0f);
                for (int column = node; column < 6; ++column) {
                    const float factor = free_inverse_factor.data[factor_base + column * 6 + node];
#pragma unroll
                    for (int row = 0; row < 3; ++row) {
                        if (row < row_count) response.c[row] += factor * free_factor[row * $PLANE + column];
                    }
                }
#pragma unroll
                for (int row = 0; row < 3; ++row) {
                    if (row >= row_count) continue;
                    const int response_offset = (row_offset + row) * support_capacity;
                    const float value = jacobian[row * $PLANE + node];
                    row_dof.data[response_offset + count + node] = offset + node;
                    row_factor.data[response_offset + count + node] = value;
                    row_free_response.data[(row_offset + row) * row_free_response.shape[2] + count + node] = response.c[row];
                    norm.c[row] += value * response.c[row];
                }
            }
            count += 6;
        } else {
$FACTOR_GROUP$MASK_ENDPOINT            for (int node = lane; node < $N; node += stride) {
$MASK_SKIP                const int physical = $PERMUTATION[node];
                const auto motion = joint_S_s.data[start + physical];
                const float velocity = v_hat.data[start + physical];
#pragma unroll
                for (int row = 0; row < 3; ++row) {
                    if (row >= row_count) continue;
                    float value = 0.0f;
                    if (art == art_a && $MASK_HAS_A)
                        value += jacobian_value(motion, angular_a_shared, row);
                    if (art == art_b && $MASK_HAS_B)
                        value -= jacobian_value(motion, angular_b_shared, row);
                    jacobian[row * $PLANE + node] = value;
                    incident.c[row] += value * velocity;
                }
            }
#if defined(__CUDA_ARCH__)
            __syncwarp(MASK);
#endif
            const int factor_base = $FACTOR_BASE;
            for (int node = lane; node < $N; node += stride) {
$MASK_SKIP                auto value = wp::vec3(0.0f);
                const int begin = $ROW_OFFSETS[node];
                const int first = node - ($ROW_OFFSETS[node + 1] - begin) + 1;
                // Reverse DFS packs this row's columns into [first, node].
$MASK_WALK_BEGIN                    const int entry = begin + column - first;
                    const float factor = $INVERSE_FACTOR[factor_base + entry];
#pragma unroll
                    for (int row = 0; row < 3; ++row) {
                        if (row < row_count) value.c[row] += factor * jacobian[row * $PLANE + column];
                    }
$MASK_WALK_END                }
                const int index = count + $MASK_RANK;
#pragma unroll
                for (int row = 0; row < 3; ++row) {
                    if (row >= row_count) continue;
                    const int response_offset = (row_offset + row) * support_capacity;
                    row_dof.data[response_offset + index] = offset + node;
                    row_factor.data[response_offset + index] = value.c[row];
                    norm.c[row] += value.c[row] * value.c[row];
                }
            }
            count += $MASK_COUNT;
        }
        // The next endpoint reuses the same group-local Jacobian storage.
#if defined(__CUDA_ARCH__)
        __syncwarp(MASK);
#endif
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        if (row >= row_count) continue;
        const int response_offset = (row_offset + row) * support_capacity;
        for (int index = count + lane; index < support_capacity; index += stride) {
            row_dof.data[response_offset + index] = -1;
            row_factor.data[response_offset + index] = 0.0f;
        }
#if defined(__CUDA_ARCH__)
        for (int shift = $LANES / 2; shift > 0; shift >>= 1) {
            norm.c[row] += __shfl_down_sync(MASK, norm.c[row], shift, $LANES);
            incident.c[row] += __shfl_down_sync(MASK, incident.c[row], shift, $LANES);
        }
#endif
        if (lane == 0) {
            row_incident.data[row_offset + row] = incident.c[row];
            diagonal.data[row_offset + row] = norm.c[row];
        }
    }
"""

# The contact response's mask sites (:func:`_expand_masks`): its endpoints' masks mask_a and mask_b,
# the endpoint's mask, the coordinate loops' membership test, the walk of a factor row's supporting
# columns, the coordinate's rank (its entry in the row) and the endpoint's coordinates.
_RESPONSE_MASK_SITES = {
    "$MASK_HELPERS": (
        """    const auto bit_count = [](unsigned long long mask) {
#if defined(__CUDA_ARCH__)
        return __popcll(mask);
#else
        int count = 0;
        while (mask != 0) { mask &= mask - 1; ++count; }
        return count;
#endif
    };
    const auto first_bit = [](unsigned long long mask) {
#if defined(__CUDA_ARCH__)
        return __ffsll(mask) - 1;
#else
        int column = 0;
        while ((mask & 1ull) == 0) { mask >>= 1; ++column; }
        return column;
#endif
    };
""",
        _MASK_HELPERS_CUDA,
    ),
    "$MASK_LOADS": (
        """    unsigned long long mask_a = 0, mask_b = 0;
    if (body_a >= 0 && art_a >= 0) mask_a = body_dof_mask.data[body_a];
    if (body_b >= 0 && art_b >= 0) mask_b = body_dof_mask.data[body_b];
""",
        """    unsigned long long mask_a[MASK_WORDS] = {}, mask_b[MASK_WORDS] = {};
    for (int word = 0; word < MASK_WORDS; ++word) {
        if (body_a >= 0 && art_a >= 0) mask_a[word] = body_dof_mask.data[body_a * MASK_WORDS + word];
        if (body_b >= 0 && art_b >= 0) mask_b[word] = body_dof_mask.data[body_b * MASK_WORDS + word];
    }
""",
    ),
    "$MASK_ENDPOINT": (
        """            unsigned long long mask = 0;
            if (art == art_a) mask |= mask_a;
            if (art == art_b) mask |= mask_b;
""",
        """            unsigned long long mask[MASK_WORDS];
            for (int word = 0; word < MASK_WORDS; ++word)
                mask[word] = (art == art_a ? mask_a[word] : 0ull) | (art == art_b ? mask_b[word] : 0ull);
""",
    ),
    "$MASK_SKIP": (
        """                const unsigned long long bit = 1ull << node;
                if ((mask & bit) == 0) continue;
""",
        """                if (!mask_test(mask, node)) continue;
""",
    ),
    "$MASK_HAS_A": ("(mask_a & bit) != 0", "mask_test(mask_a, node)"),
    "$MASK_HAS_B": ("(mask_b & bit) != 0", "mask_test(mask_b, node)"),
    "$MASK_WALK_BEGIN": (
        """                // Visit only supporting columns, including bit 63 without a shift by 64.
                unsigned long long pending = mask & (~0ull << first) & (~0ull >> (63 - node));
                while (pending != 0) {
                    const int column = first_bit(pending);
""",
        """                // Visit only supporting columns, word by word, without a shift by 64.
                unsigned long long pending[MASK_WORDS];
                for (int word = 0; word < MASK_WORDS; ++word) {
                    const int low = first - 64 * word;
                    const int high = node - 64 * word;
                    pending[word] = low > 63 || high < 0 ? 0ull
                        : mask[word] & (low > 0 ? ~0ull << low : ~0ull) & (high < 63 ? ~0ull >> (63 - high) : ~0ull);
                }
                for (int column = mask_next(pending); column >= 0; column = mask_next(pending)) {
""",
    ),
    "$MASK_WALK_END": (
        """                    pending &= pending - 1;
""",
        "",
    ),
    "$MASK_RANK": ("bit_count(mask & (bit - 1ull))", "mask_rank(mask, node)"),
    "$MASK_COUNT": ("bit_count(mask)", "mask_rank(mask, 64 * MASK_WORDS)"),
}

# The specialised kernel's factor branch: one plan, whose arrays are kernel inputs, and N constant.
_SPECIALIZED_FACTOR = {
    "$FACTOR_GROUP": "",
    "$PERMUTATION": "permutation.data",
    "$ROW_OFFSETS": "factor_row_offsets.data",
    "$FACTOR_BASE": "group * inverse_factor.shape[1]",
    "$INVERSE_FACTOR": "inverse_factor.data",
}

# The group-table kernel's factor branch: the endpoint's group gives N, its plan's bases in the
# concatenated permutation and row offsets, and its factor's first entry in the flat Linv.
_GROUP_TABLE_FACTOR = {
    "$FACTOR_GROUP": """            const int factor_group = art_group.data[art];
            const int dof_count = group_dof_count.data[factor_group];
            const int* factor_permutation = permutation.data + group_permutation_base.data[factor_group];
            const int* factor_row_offsets = row_offsets.data + group_row_offset_base.data[factor_group];
""",
    "$PERMUTATION": "factor_permutation",
    "$ROW_OFFSETS": "factor_row_offsets",
    "$FACTOR_BASE": (
        "group_linv_offset.data[factor_group] + art_slot.data[art] * group_nonzero_count.data[factor_group]"
    ),
    "$INVERSE_FACTOR": "Linv.data",
    "$N": "dof_count",
}


def _check_response_lanes(lanes: int, warps: int) -> None:
    if lanes not in (8, 16, 32) or not 1 <= warps <= 32:
        raise ValueError("Invalid sparse contact dimensions or lane group")


def _response_storage_floats(scratch_dofs: int, lanes: int, warps: int) -> int:
    """Floats of :data:`_RESPONSE_SNIPPET`'s one static shared array, ``storage`` (``$STORAGE``).

    Each lane group of the block keeps three rows, one per contact direction, of
    ``max(scratch_dofs, 6)`` Jacobian entries and the free factor's 6.
    """
    groups = warps * 32 // lanes
    return groups * 3 * (max(scratch_dofs, 6) + 6)


WARP_SHARED_BYTES_PER_THREAD = 4
"""Static shared memory Warp itself declares in every kernel, per thread of the block it is built for.

It is the tile storage's ``unsigned int smem_base[WP_TILE_BLOCK_DIM]`` (``tile_shared_storage_t``,
Warp 1.18 and 1.19), so a kernel's own shared arrays get a block's static shared memory less
``4 * block_dim`` bytes.
"""


def scene_contact_response_shared_bytes(dof_max: int, *, lanes_per_contact: int = 8, warps_per_block: int = 4) -> int:
    """Bytes of static shared memory per block of :func:`_get_scene_contact_response_kernel` for these sizes.

    The snippet's ``storage`` array (:func:`_response_storage_floats`), which grows with ``dof_max``
    and not with the mask words, plus Warp's reservation (:data:`WARP_SHARED_BYTES_PER_THREAD`) for
    the ``32 * warps_per_block`` threads of the kernel's launch. A CUDA block holds at most 48 KiB of
    static shared memory, so the scene solve checks this before it builds the kernel.
    """
    lanes, warps = int(lanes_per_contact), int(warps_per_block)
    _check_response_lanes(lanes, warps)
    return 4 * _response_storage_floats(int(dof_max), lanes, warps) + WARP_SHARED_BYTES_PER_THREAD * 32 * warps


def scene_contact_response_dof_limit(shared_bytes: int, *, lanes_per_contact: int = 8, warps_per_block: int = 4) -> int:
    """The most DOFs whose :func:`scene_contact_response_shared_bytes` fits ``shared_bytes``; 0 when none does."""
    sizes = {"lanes_per_contact": lanes_per_contact, "warps_per_block": warps_per_block}
    if scene_contact_response_shared_bytes(1, **sizes) > shared_bytes:
        return 0
    # The storage grows with dof_max: double past the limit, then bisect.
    low, high = 1, 2
    while scene_contact_response_shared_bytes(high, **sizes) <= shared_bytes:
        low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        if scene_contact_response_shared_bytes(middle, **sizes) <= shared_bytes:
            low = middle
        else:
            high = middle
    return low


def _response_snippet(
    scratch_dofs: int, *, lanes: int = 8, warps: int = 4, mask_words: int = 1, specialized: bool = False
) -> str:
    """:data:`_RESPONSE_SNIPPET` with scratch for ``scratch_dofs`` coordinates, support masks of ``mask_words`` words.

    ``specialized`` fills the factor branch of Steel's specialised kernel (:data:`_SPECIALIZED_FACTOR`, one plan of
    ``scratch_dofs`` coordinates): with one mask word it is the snippet of
    :func:`~newton._src.solvers.steel.sparse_contact._get_sparse_contact_response_kernel` for ``scratch_dofs`` DOFs,
    character for character. Otherwise the factor branch reads the endpoint's plan through the scene's group table
    (:data:`_GROUP_TABLE_FACTOR`).
    """
    scratch_size = max(scratch_dofs, 6) + 6
    factor = {**_SPECIALIZED_FACTOR, "$N": str(scratch_dofs)} if specialized else _GROUP_TABLE_FACTOR
    snippet = _expand_masks(_RESPONSE_SNIPPET, _RESPONSE_MASK_SITES, mask_words)
    for placeholder, text in factor.items():
        snippet = snippet.replace(placeholder, text)
    return (
        snippet.replace("$STORAGE", str(_response_storage_floats(scratch_dofs, lanes, warps)))
        .replace("$LANES", str(lanes))
        .replace("$SCRATCH", str(3 * scratch_size))
        .replace("$PLANE", str(scratch_size))
        .replace("$JACOBIAN", str(max(scratch_dofs, 6)))
    )


def scene_contact_response_snippet(
    dof_max: int, mask_words: int = 1, *, lanes_per_contact: int = 8, warps_per_block: int = 4
) -> str:
    """The native snippet of :func:`_get_scene_contact_response_kernel` for these sizes.

    It is Steel's specialised snippet (:func:`_response_snippet` with ``specialized``), line for line, except
    that the factor branch
    reads the endpoint's plan and factor through the group table and its scratch holds ``dof_max``
    coordinates. Its support masks have ``mask_words`` ``uint64`` words (``body_dof_mask`` holds
    ``mask_words`` consecutive words per body); with more than one, the mask lines call the helpers
    of :data:`_MASK_HELPERS_CUDA`, and with one they are the single-word source, character for
    character. Raises :class:`ValueError` for the sizes the kernel rejects: ``dof_max`` past
    ``64 * mask_words`` among them.
    """
    n, words, lanes, warps = int(dof_max), int(mask_words), int(lanes_per_contact), int(warps_per_block)
    if words < 1 or not 1 <= n <= 64 * words:
        raise ValueError("Invalid sparse contact dimensions or lane group")
    _check_response_lanes(lanes, warps)
    return _response_snippet(n, lanes=lanes, warps=warps, mask_words=words)


@cache
def _get_scene_contact_response_kernel(
    dof_max: int,
    mask_words: int = 1,
    *,
    lanes_per_contact: int = 8,
    warps_per_block: int = 4,
    slots_needed: bool = False,
    patches: bool = False,
) -> wp.Kernel:
    """Project contact rows as Steel's
    :func:`~newton._src.solvers.steel.sparse_contact._get_sparse_contact_response_kernel` does, for any factor
    group.

    The arguments are the specialised kernel's, with its one plan (``permutation``,
    ``factor_row_offsets``, ``inverse_factor``) replaced by a group table of the scene's factor
    groups (:class:`~newton._src.solvers.steel.scene_solve.SceneGroup`):

    - ``art_group`` [articulations]: the articulation's factor group, ``-1`` for free bodies and
      articulations without a response; ``art_slot`` [articulations]: its position in the group.
    - ``group_dof_count``, ``group_nonzero_count``, ``group_linv_offset``,
      ``group_permutation_base`` and ``group_row_offset_base`` [groups]: the group's DOFs, its
      factor entries per member, its first entry in ``Linv`` and the first entries of its plan in
      ``permutation`` [sum of DOFs] and ``row_offsets`` [sum of DOFs + 1, CSR per group].
    - ``Linv`` [sum of members x nonzeros]: every group's inverse factors, flat; member ``slot`` of
      group ``g`` owns ``nonzero_count`` entries from ``linv_offset + slot * nonzero_count``.

    Per endpoint the arithmetic and its order are the specialised kernel's
    (:func:`scene_contact_response_snippet`), so a scene of one factor group gets the specialised
    kernel's rows bit for bit. The free-body branch is the same code, indexed by ``art_group_idx``.
    The shared scratch holds ``dof_max`` coordinates, the largest group's, and ``body_dof_mask``
    [bodies x ``mask_words``] each body's support mask in ``mask_words`` words, the widest group's
    (a narrower group's masks are zero in the words past its own). Launch as the specialised kernel.

    Every contact has three rows, as in Steel's sparse path, unless ``slots_needed``: then the
    kernel takes ``contact_slots_needed`` after ``contact_path`` and writes each contact's rows
    from it, one (the normal row of a contact past ``contact_friction_gap_max``) or three, so a
    one-row unit's row is written alone and the next unit's rows are left to it. It also takes
    ``anchor_point0`` and ``anchor_point1`` (``wp.array[wp.vec3]``) after the shared-anchor flags,
    which only ``patches`` reads (one-element placeholders without it).

    ``patches`` (persistent friction patches) implies ``slots_needed``, a patch region's members
    past its anchors holding their normal row alone (Steel's allocation,
    ``kernels._allocate_world_contact_slot``), and reads
    ``anchor_point0`` and ``anchor_point1`` (``FrictionPatches.point_a`` and ``point_b``: each
    anchor's point in world space, the two equal). The tangent rows take them as their lever arms,
    ``wrench = cross(anchor - origin, direction)`` on each responding side, free bodies and
    articulations alike, in place of the witness points and in precedence over the shared-anchor
    flags (Steel's ``kernels.contact_row_points``); the normal row keeps the witness points,
    or their midpoint under ``contact_shared_anchor``. It is the ``slots_needed`` definition, its
    tangent rows' points selected by ``wp.static(patches)``, under its own key (``_slots_patches``);
    the native snippet is the default's, and the default definition is unchanged.
    """
    if patches and not slots_needed:
        return _get_scene_contact_response_kernel(
            dof_max,
            mask_words,
            lanes_per_contact=lanes_per_contact,
            warps_per_block=warps_per_block,
            slots_needed=True,
            patches=True,
        )
    snippet = scene_contact_response_snippet(
        dof_max, mask_words, lanes_per_contact=lanes_per_contact, warps_per_block=warps_per_block
    )
    n, words, lanes, warps = int(dof_max), int(mask_words), int(lanes_per_contact), int(warps_per_block)

    @wp.func_native(snippet)
    def project_native(
        tid: int,
        body_a: int,
        body_b: int,
        art_a: int,
        art_b: int,
        world: int,
        slot: int,
        row_count: int,
        angular_a: wp.mat33,
        angular_b: wp.mat33,
        directions: wp.mat33,
        art_group_idx: wp.array[int],
        articulation_response_dof_count: wp.array[int],
        is_free_rigid: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        body_dof_mask: wp.array[wp.uint64],
        joint_S_s: wp.array[wp.spatial_vector],
        art_group: wp.array[int],
        art_slot: wp.array[int],
        group_dof_count: wp.array[int],
        group_nonzero_count: wp.array[int],
        group_linv_offset: wp.array[int],
        group_permutation_base: wp.array[int],
        group_row_offset_base: wp.array[int],
        permutation: wp.array[int],
        row_offsets: wp.array[int],
        Linv: wp.array[float],
        free_inverse_factor: wp.array3d[float],
        v_hat: wp.array[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
    ): ...

    if slots_needed:

        def contact_rows_kernel(  # pragma: no cover - compiled by wp.kernel
            contact_count: wp.array[int],
            total_num_workers: int,
            contact_point0: wp.array[wp.vec3],
            contact_point1: wp.array[wp.vec3],
            contact_normal: wp.array[wp.vec3],
            contact_shape0: wp.array[int],
            contact_shape1: wp.array[int],
            contact_thickness0: wp.array[float],
            contact_thickness1: wp.array[float],
            contact_world: wp.array[int],
            contact_slot: wp.array[int],
            contact_art_a: wp.array[int],
            contact_art_b: wp.array[int],
            contact_path: wp.array[int],
            contact_slots_needed: wp.array[int],
            art_group_idx: wp.array[int],
            articulation_response_dof_count: wp.array[int],
            is_free_rigid: wp.array[int],
            articulation_world_dof_offset: wp.array[int],
            articulation_dof_start: wp.array[int],
            articulation_origin: wp.array[wp.vec3],
            body_dof_mask: wp.array[wp.uint64],
            joint_S_s: wp.array[wp.spatial_vector],
            shape_body: wp.array[int],
            body_q: wp.array[wp.transform],
            contact_shared_anchor: int,
            contact_friction_shared_anchor: int,
            anchor_point0: wp.array[wp.vec3],
            anchor_point1: wp.array[wp.vec3],
            art_group: wp.array[int],
            art_slot: wp.array[int],
            group_dof_count: wp.array[int],
            group_nonzero_count: wp.array[int],
            group_linv_offset: wp.array[int],
            group_permutation_base: wp.array[int],
            group_row_offset_base: wp.array[int],
            permutation: wp.array[int],
            row_offsets: wp.array[int],
            Linv: wp.array[float],
            free_inverse_factor: wp.array3d[float],
            v_hat: wp.array[float],
            row_dof: wp.array3d[int],
            row_factor: wp.array3d[float],
            row_free_response: wp.array3d[float],
            row_incident: wp.array2d[float],
            diagonal: wp.array2d[float],
        ):
            tid = wp.tid()
            worker = tid // lanes
            total_contacts = wp.min(contact_count[0], contact_point0.shape[0])
            for c in range(worker, total_contacts, total_num_workers):
                # A contact owns its normal row and, within the friction gap (as a patch anchor under patches), two
                # friction rows.
                row_count = contact_slots_needed[c]
                if contact_path[c] != 0 or contact_slot[c] < 0:
                    continue
                body_a = int(-1)
                body_b = int(-1)
                if contact_shape0[c] >= 0:
                    body_a = shape_body[contact_shape0[c]]
                if contact_shape1[c] >= 0:
                    body_b = shape_body[contact_shape1[c]]
                art_a = contact_art_a[c]
                art_b = contact_art_b[c]
                angular_a = wp.mat33(0.0)
                angular_b = wp.mat33(0.0)
                directions = wp.mat33(0.0)
                if tid % lanes == 0:
                    normal = -contact_normal[c]
                    point_a = contact_point0[c] - contact_thickness0[c] * normal
                    point_b = contact_point1[c] + contact_thickness1[c] * normal
                    if body_a >= 0:
                        point_a = wp.transform_point(body_q[body_a], contact_point0[c]) - contact_thickness0[c] * normal
                    if body_b >= 0:
                        point_b = wp.transform_point(body_q[body_b], contact_point1[c]) + contact_thickness1[c] * normal
                    friction_point_a = point_a
                    friction_point_b = point_b
                    # Shared anchors put both bodies' points at the witness midpoint.
                    midpoint = 0.5 * (point_a + point_b)
                    if contact_shared_anchor != 0:
                        point_a = midpoint
                        point_b = midpoint
                    if wp.static(patches):
                        # The tangent rows act at the region's anchor (Steel's contact_row_points), in precedence over
                        # both flags.
                        friction_point_a = anchor_point0[c]
                        friction_point_b = anchor_point1[c]
                    elif contact_shared_anchor != 0 or contact_friction_shared_anchor != 0:
                        friction_point_a = midpoint
                        friction_point_b = midpoint
                    tangent0, tangent1 = contact_tangent_basis(normal)
                    for row in range(3):
                        direction = normal
                        row_point_a = point_a
                        row_point_b = point_b
                        if row == 1:
                            direction = tangent0
                        elif row == 2:
                            direction = tangent1
                        if row > 0:
                            row_point_a = friction_point_a
                            row_point_b = friction_point_b
                        # J = [direction, (point - origin) x direction] dot motion.
                        wrench_a = wp.vec3(0.0)
                        wrench_b = wp.vec3(0.0)
                        if art_a >= 0 and articulation_response_dof_count[art_a] > 0:
                            wrench_a = wp.cross(row_point_a - articulation_origin[art_a], direction)
                        if art_b >= 0 and articulation_response_dof_count[art_b] > 0:
                            wrench_b = wp.cross(row_point_b - articulation_origin[art_b], direction)
                        for axis in range(3):
                            angular_a[row, axis] = wrench_a[axis]
                            angular_b[row, axis] = wrench_b[axis]
                            directions[row, axis] = direction[axis]
                project_native(
                    tid,
                    body_a,
                    body_b,
                    art_a,
                    art_b,
                    contact_world[c],
                    contact_slot[c],
                    row_count,
                    angular_a,
                    angular_b,
                    directions,
                    art_group_idx,
                    articulation_response_dof_count,
                    is_free_rigid,
                    articulation_world_dof_offset,
                    articulation_dof_start,
                    body_dof_mask,
                    joint_S_s,
                    art_group,
                    art_slot,
                    group_dof_count,
                    group_nonzero_count,
                    group_linv_offset,
                    group_permutation_base,
                    group_row_offset_base,
                    permutation,
                    row_offsets,
                    Linv,
                    free_inverse_factor,
                    v_hat,
                    row_dof,
                    row_factor,
                    row_free_response,
                    row_incident,
                    diagonal,
                )

        contact_rows_kernel.__name__ = (
            f"populate_scene_contact_response_{n}_{lanes}_{warps}"
            + (f"_w{words}" if words > 1 else "")
            + ("_slots_patches" if patches else "_slots")
        )
        contact_rows_kernel.__qualname__ = contact_rows_kernel.__name__
        return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(
            contact_rows_kernel
        )

    def contact_kernel(  # pragma: no cover - compiled by wp.kernel
        contact_count: wp.array[int],
        total_num_workers: int,
        contact_point0: wp.array[wp.vec3],
        contact_point1: wp.array[wp.vec3],
        contact_normal: wp.array[wp.vec3],
        contact_shape0: wp.array[int],
        contact_shape1: wp.array[int],
        contact_thickness0: wp.array[float],
        contact_thickness1: wp.array[float],
        contact_world: wp.array[int],
        contact_slot: wp.array[int],
        contact_art_a: wp.array[int],
        contact_art_b: wp.array[int],
        contact_path: wp.array[int],
        art_group_idx: wp.array[int],
        articulation_response_dof_count: wp.array[int],
        is_free_rigid: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        articulation_origin: wp.array[wp.vec3],
        body_dof_mask: wp.array[wp.uint64],
        joint_S_s: wp.array[wp.spatial_vector],
        shape_body: wp.array[int],
        body_q: wp.array[wp.transform],
        contact_shared_anchor: int,
        contact_friction_shared_anchor: int,
        art_group: wp.array[int],
        art_slot: wp.array[int],
        group_dof_count: wp.array[int],
        group_nonzero_count: wp.array[int],
        group_linv_offset: wp.array[int],
        group_permutation_base: wp.array[int],
        group_row_offset_base: wp.array[int],
        permutation: wp.array[int],
        row_offsets: wp.array[int],
        Linv: wp.array[float],
        free_inverse_factor: wp.array3d[float],
        v_hat: wp.array[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
    ):
        tid = wp.tid()
        worker = tid // lanes
        total_contacts = wp.min(contact_count[0], contact_point0.shape[0])
        for c in range(worker, total_contacts, total_num_workers):
            # Every dense contact owns a normal row and two friction rows.
            row_count = 3
            if contact_path[c] != 0 or contact_slot[c] < 0:
                continue
            body_a = int(-1)
            body_b = int(-1)
            if contact_shape0[c] >= 0:
                body_a = shape_body[contact_shape0[c]]
            if contact_shape1[c] >= 0:
                body_b = shape_body[contact_shape1[c]]
            art_a = contact_art_a[c]
            art_b = contact_art_b[c]
            angular_a = wp.mat33(0.0)
            angular_b = wp.mat33(0.0)
            directions = wp.mat33(0.0)
            if tid % lanes == 0:
                normal = -contact_normal[c]
                point_a = contact_point0[c] - contact_thickness0[c] * normal
                point_b = contact_point1[c] + contact_thickness1[c] * normal
                if body_a >= 0:
                    point_a = wp.transform_point(body_q[body_a], contact_point0[c]) - contact_thickness0[c] * normal
                if body_b >= 0:
                    point_b = wp.transform_point(body_q[body_b], contact_point1[c]) + contact_thickness1[c] * normal
                friction_point_a = point_a
                friction_point_b = point_b
                # Shared anchors put both bodies' points at the witness midpoint.
                midpoint = 0.5 * (point_a + point_b)
                if contact_shared_anchor != 0:
                    point_a = midpoint
                    point_b = midpoint
                if contact_shared_anchor != 0 or contact_friction_shared_anchor != 0:
                    friction_point_a = midpoint
                    friction_point_b = midpoint
                tangent0, tangent1 = contact_tangent_basis(normal)
                for row in range(3):
                    direction = normal
                    row_point_a = point_a
                    row_point_b = point_b
                    if row == 1:
                        direction = tangent0
                    elif row == 2:
                        direction = tangent1
                    if row > 0:
                        row_point_a = friction_point_a
                        row_point_b = friction_point_b
                    # J = [direction, (point - origin) x direction] dot motion.
                    wrench_a = wp.vec3(0.0)
                    wrench_b = wp.vec3(0.0)
                    if art_a >= 0 and articulation_response_dof_count[art_a] > 0:
                        wrench_a = wp.cross(row_point_a - articulation_origin[art_a], direction)
                    if art_b >= 0 and articulation_response_dof_count[art_b] > 0:
                        wrench_b = wp.cross(row_point_b - articulation_origin[art_b], direction)
                    for axis in range(3):
                        angular_a[row, axis] = wrench_a[axis]
                        angular_b[row, axis] = wrench_b[axis]
                        directions[row, axis] = direction[axis]
            project_native(
                tid,
                body_a,
                body_b,
                art_a,
                art_b,
                contact_world[c],
                contact_slot[c],
                row_count,
                angular_a,
                angular_b,
                directions,
                art_group_idx,
                articulation_response_dof_count,
                is_free_rigid,
                articulation_world_dof_offset,
                articulation_dof_start,
                body_dof_mask,
                joint_S_s,
                art_group,
                art_slot,
                group_dof_count,
                group_nonzero_count,
                group_linv_offset,
                group_permutation_base,
                group_row_offset_base,
                permutation,
                row_offsets,
                Linv,
                free_inverse_factor,
                v_hat,
                row_dof,
                row_factor,
                row_free_response,
                row_incident,
                diagonal,
            )

    contact_kernel.__name__ = f"populate_scene_contact_response_{n}_{lanes}_{warps}" + (
        f"_w{words}" if words > 1 else ""
    )
    contact_kernel.__qualname__ = contact_kernel.__name__
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(contact_kernel)


# The internal-row builders' one native body, one warp per articulation of a factor group: the
# candidates of a row family and its law, the reservation (COUNT, WRITE or ATOMIC, the cursor contract
# that scene_assign_rows and the dropped units rely on), the ballot visit in lane order with its
# row-capacity break, the sparse response and the row's tail. A row family fills its family sites
# ($FAMILY_*, :func:`_expand_family`; _LIMIT_FAMILY_SITES, _FRICTION_FAMILY_SITES, _DRIVE_FAMILY_SITES,
# _FUSED_DRIVE_FAMILY_SITES, _VELOCITY_LIMIT_FAMILY_SITES, _MIMIC_FAMILY_SITES) and every family the mask sites of
# _JOINT_DOF_ROWS_MASK_SITES, but the mimic family, whose support mask is its two DOFs' (_MIMIC_MASK_SITES). The
# joint-limit expansion is Steel's builder's source. The cursor contract
# serves any per-articulation unit: the internal unit's families advance its cursor, the velocity-limit
# family the late unit's (scene_solve.SceneSolve.launch_late_rows).
_JOINT_DOF_ROWS_TEMPLATE = """
    const int group = tid >> 5;
    if (group >= group_to_art.shape[0]) return;
#if defined(__CUDA_ARCH__)
    constexpr unsigned MASK = 0xffffffffu;
    const int lane = tid & 31;
    const int width = 32;
#else
    if ((tid & 31) != 0) return;
    const int lane = 0;
    const int width = 1;
#endif
$MASK_HELPERS    const int art = group_to_art.data[group];
$FAMILY_ART_SKIP    const int world = art_to_world.data[art];
    const int start = articulation_dof_start.data[art];
    const int dofs = inverse_permutation.shape[0];
    const int capacity = row_type.shape[1];
    const int support_capacity = row_dof.shape[2];
    const int factor_base = group * inverse_factor.shape[1];
$FAMILY_CANDIDATE_HEAD        const int dof = start + local_dof;
$FAMILY_CANDIDATE_VALUES        int active = 0;
$FAMILY_LAW#if defined(__CUDA_ARCH__)
        const unsigned active_mask = __ballot_sync(MASK, active != 0);
        const int active_count = __popc(active_mask);
        int first_slot = 0;
        if (lane == 0 && active_count != 0) {
            if (row_slot_mode == $ROW_SLOTS_ATOMIC) {
                first_slot = atomicAdd(&slot_counter.data[world], active_count);
            } else {
                // One warp owns the articulation; see reserve_world_row_slots.
                first_slot = element_row_cursor.data[art];
                element_row_cursor.data[art] = first_slot + active_count;
            }
        }
        first_slot = __shfl_sync(MASK, first_slot, 0);
#else
        const unsigned active_mask = active != 0 ? 1u : 0u;
        int first_slot = 0;
        if (row_slot_mode == $ROW_SLOTS_ATOMIC) {
            first_slot = slot_counter.data[world];
            slot_counter.data[world] += active;
        } else {
            first_slot = element_row_cursor.data[art];
            element_row_cursor.data[art] = first_slot + active;
        }
#endif
$FAMILY_VISIT        unsigned pending = row_slot_mode == $ROW_SLOTS_COUNT ? 0u : active_mask;
        int rank = 0;
        while (pending != 0) {
            const int row = first_slot + rank;
            // The reservation above still counts every over-capacity row.
            if (row >= capacity) break;
#if defined(__CUDA_ARCH__)
            const int source = __ffs(pending) - 1;
            const int selected_dof = __shfl_sync(MASK, local_dof, source);
$FAMILY_SELECT_CUDA#else
            const int selected_dof = local_dof;
$FAMILY_SELECT_CPU#endif
$FAMILY_SIGN_DECL            const int column = inverse_permutation.data[selected_dof];
$MASK_LOAD            const int row_offset = world * capacity + row;
            const int response_offset = row_offset * support_capacity;
            float norm = 0.0f;
            for (int node = lane; node < dofs; node += width) {
$MASK_SKIP                const int index = $MASK_RANK;
                const int entry = factor_lookup.data[node * dofs + column];
$FAMILY_VALUE                row_dof.data[response_offset + index] = articulation_world_dof_offset.data[art] + node;
                row_factor.data[response_offset + index] = value;
                norm += value * value;
            }
            const int support_count = $MASK_COUNT;
            for (int index = support_count + lane; index < support_capacity; index += width) {
                row_dof.data[response_offset + index] = -1;
                row_factor.data[response_offset + index] = 0.0f;
            }
#if defined(__CUDA_ARCH__)
            for (int shift = 16; shift > 0; shift >>= 1) norm += __shfl_down_sync(MASK, norm, shift);
#endif
            if (lane == 0) {
                row_incident.data[row_offset] = $FAMILY_INCIDENT;
                diagonal.data[row_offset] = norm;
                row_type.data[row_offset] = $FAMILY_ROW_TYPE;
                row_parent.data[row_offset] = -1;
$FAMILY_ROW_TAIL            }
            pending &= pending - 1;
            ++rank;
        }
    }
"""

# The joint-limit family's sites: two candidates per DOF (lower then upper bound), Steel's limit law
# (active within the activation gap of a finite bound, ``phi`` the gap), and the response of
# ``J = +1`` on the DOF for the lower bound and ``-1`` for the upper.
_LIMIT_FAMILY_SITES = {
    "$FAMILY_ART_SKIP": "",
    "$FAMILY_CANDIDATE_HEAD": """\
    for (int base = 0; base < 2 * dofs; base += width) {
        const int candidate = base + lane;
        const int local_dof = candidate >> 1;
        const int side = candidate & 1;
""",
    "$FAMILY_CANDIDATE_VALUES": """\
        const int q_index = candidate < 2 * dofs ? limit_q_index.data[dof] : -1;
        float gap = 0.0f;
""",
    "$FAMILY_LAW": """\
        if (q_index >= 0) {
            const float position = joint_q.data[q_index];
            const float bound = side == 0 ? joint_limit_lower.data[dof] : joint_limit_upper.data[dof];
            gap = side == 0 ? position - bound : bound - position;
            active = wp::isfinite(bound)
                && (side == 0 ? position <= bound + activation_gap : position >= bound - activation_gap);
        }
""",
    "$FAMILY_VISIT": """\
        // Visit the ballot in lane order, matching the existing lower/upper DOF order.
        // A COUNT pass only advances the cursor.
""",
    "$FAMILY_SELECT_CUDA": """\
            const int selected_side = __shfl_sync(MASK, side, source);
            const float selected_gap = __shfl_sync(MASK, gap, source);
""",
    "$FAMILY_SELECT_CPU": """\
            const int selected_side = side;
            const float selected_gap = gap;
""",
    "$FAMILY_SIGN_DECL": """\
            const float sign = selected_side == 0 ? 1.0f : -1.0f;
""",
    "$FAMILY_VALUE": """\
                const float value = entry >= 0 ? sign * inverse_factor.data[factor_base + entry] : 0.0f;
""",
    "$FAMILY_INCIDENT": "sign * v_hat.data[start + selected_dof]",
    "$FAMILY_ROW_TYPE": str(PGS_CONSTRAINT_TYPE_JOINT_LIMIT),
    "$FAMILY_ROW_TAIL": """\
                row_mu.data[row_offset] = 0.0f;
                phi.data[row_offset] = selected_gap;
                target_velocity.data[row_offset] = 0.0f;
""",
}

# The joint-friction family's sites: Steel's joint friction law (solver_steel._get_joint_friction_warp_kernel): an
# active articulation's supported, non-kinematic DOF with friction f builds a row of J = 1 on the DOF and bound f dt
# when f dt > 0 and f is finite, one candidate per DOF.
_FRICTION_FAMILY_SITES = {
    "$FAMILY_ART_SKIP": "    if (articulation_rows_active.data[art] == 0) return;\n",
    "$FAMILY_CANDIDATE_HEAD": """\
    for (int base = 0; base < dofs; base += width) {
        const int local_dof = base + lane;
""",
    "$FAMILY_CANDIDATE_VALUES": """\
        float bound = 0.0f;
""",
    "$FAMILY_LAW": """\
        if (local_dof < dofs && friction_dof_supported.data[dof] != 0 && kinematic_dof_mask.data[dof] == 0) {
            // Honor the stride: callers may replace Model.joint_friction with a strided view.
            const float friction = *reinterpret_cast<const float*>(
                reinterpret_cast<const char*>(joint_friction.data) + dof * joint_friction.strides[0]);
            bound = friction * dt;
            // Non-finite or negative coefficients build no row; the per-step check flags them.
            active = bound > 0.0f && wp::isfinite(friction);
        }
""",
    "$FAMILY_VISIT": """\
        // Visit the ballot in lane order, the articulation's DOF order. A COUNT pass only advances the cursor.
""",
    "$FAMILY_SELECT_CUDA": """\
            const float selected_bound = __shfl_sync(MASK, bound, source);
""",
    "$FAMILY_SELECT_CPU": """\
            const float selected_bound = bound;
""",
    "$FAMILY_SIGN_DECL": "",
    "$FAMILY_VALUE": """\
                // J = 1 on the DOF (Steel's J_group entry), so the response is the factor column itself.
                const float value = entry >= 0 ? inverse_factor.data[factor_base + entry] : 0.0f;
""",
    "$FAMILY_INCIDENT": "v_hat.data[start + selected_dof]",
    "$FAMILY_ROW_TYPE": str(PGS_CONSTRAINT_TYPE_JOINT_FRICTION),
    "$FAMILY_ROW_TAIL": """\
                // Steel keeps the bound in mu; the scene's sweeps read it from row_bound.
                row_mu.data[row_offset] = selected_bound;
                phi.data[row_offset] = 0.0f;
                target_velocity.data[row_offset] = 0.0f;
                row_bound.data[row_offset] = selected_bound;
""",
}


# The drive family's sites: Steel's drive rows (kernels.allocate_physx_drive_slots reserves a row
# for every DOF of a PRISMATIC, REVOLUTE or D6 joint whose target stiffness or damping is positive, and
# kernels.populate_physx_drive_J_for_size fills it), one candidate per DOF. A row has J = 1 on the DOF, no mu or phi,
# the DOF's target velocity, and the parameters its descriptors read (scene_solve.scene_drive_descriptors):
# stiffness, damping, the position error target - q and the effort limit. limit_q_index holds each DOF's
# coordinate, -1 outside those three joint types, which is Steel's joint-type test and its target index
# joint_target_q_start[j] + axis (joint_target_q_start is joint_q_start). The model and control arrays are
# read through their strides, as Steel's kernels index them.
_DRIVE_FAMILY_SITES = {
    "$FAMILY_ART_SKIP": "",
    "$FAMILY_CANDIDATE_HEAD": """\
    for (int base = 0; base < dofs; base += width) {
        const int local_dof = base + lane;
""",
    "$FAMILY_CANDIDATE_VALUES": """\
        const int q_index = local_dof < dofs ? limit_q_index.data[dof] : -1;
""",
    "$FAMILY_LAW": """\
        if (q_index >= 0) {
            // Steel skips a DOF whose target stiffness and damping are both non-positive.
            active = !(wp::index(joint_target_ke, dof) <= 0.0f && wp::index(joint_target_kd, dof) <= 0.0f);
        }
""",
    "$FAMILY_VISIT": """\
        // Visit the ballot in lane order, the articulation's DOF order. A COUNT pass only advances the cursor.
""",
    "$FAMILY_SELECT_CUDA": "",
    "$FAMILY_SELECT_CPU": "",
    "$FAMILY_SIGN_DECL": "",
    "$FAMILY_VALUE": """\
                // J = 1 on the DOF (Steel's J_group entry), so the response is the factor column itself.
                const float value = entry >= 0 ? inverse_factor.data[factor_base + entry] : 0.0f;
""",
    "$FAMILY_INCIDENT": "v_hat.data[start + selected_dof]",
    "$FAMILY_ROW_TYPE": str(PGS_CONSTRAINT_TYPE_JOINT_TARGET),
    "$FAMILY_ROW_TAIL": """\
                // Steel's drive row: its target velocity and the drive parameters of its DOF.
                const int drive_dof = start + selected_dof;
                const int drive_q = limit_q_index.data[drive_dof];
                row_mu.data[row_offset] = 0.0f;
                phi.data[row_offset] = 0.0f;
                target_velocity.data[row_offset] = wp::index(joint_target_qd, drive_dof);
                drive_stiffness.data[row_offset] = wp::index(joint_target_ke, drive_dof);
                drive_damping.data[row_offset] = wp::index(joint_target_kd, drive_dof);
                drive_geom_error.data[row_offset] = wp::index(joint_target_q, drive_q) - wp::index(joint_q, drive_q);
                drive_max_force.data[row_offset] = wp::index(joint_effort_limit, drive_dof);
""",
}

# The drive family's sites with the fused velocity clamp (fuse_joint_velocity_limits with joint velocity limits):
# the row's tail also records its DOF's velocity limit, which phase C's clamp reads (Steel's
# kernels.populate_physx_drive_J_for_size: a non-positive or non-finite limit is stored as inf, which
# the clamp never passes). Every other site is the drive family's, so the unfused text is unchanged.
_FUSED_DRIVE_FAMILY_SITES = {
    **_DRIVE_FAMILY_SITES,
    "$FAMILY_ROW_TAIL": _DRIVE_FAMILY_SITES["$FAMILY_ROW_TAIL"]
    + """\
                // The fused clamp's limit: inf for a non-positive or non-finite joint velocity limit.
                const float drive_qdot_max = wp::index(joint_velocity_limit, drive_dof);
                drive_vel_limit.data[row_offset] =
                    drive_qdot_max > 0.0f && wp::isfinite(drive_qdot_max) ? drive_qdot_max : INFINITY;
""",
}


# The velocity-limit family's sites: Steel's joint velocity-limit rows (kernels.allocate_joint_velocity_limit_slots
# reserves them, kernels.populate_joint_velocity_limit_J_for_size fills them), two candidates per DOF in Steel's slot
# order, the lower bound (sign +1, J = +1) then the upper (sign -1, J = -1). A DOF of a PRISMATIC, REVOLUTE or D6 joint
# (limit_q_index >= 0, Steel's joint-type test) of an active articulation (sleeping ones reserve none) is a candidate
# unless its joint_velocity_limit qdot_max is non-positive, the fused clamp limits it (skip_driven, a drive row marked
# in drive_slot), or the activation gate holds it back (|v_hat| < fraction qdot_max with a positive fraction; a
# fraction of 0 does not read v_hat), the tests of allocate_joint_velocity_limit_slots. Steel's tests are mirrored as
# written, so a NaN limit or velocity passes them. The row has no mu, no phi and the target -qdot_max on both sides of
# the box (populate_joint_velocity_limit_J_for_size); its rows are late rows, swept in phase C.
_VELOCITY_LIMIT_FAMILY_SITES = {
    "$FAMILY_ART_SKIP": _FRICTION_FAMILY_SITES["$FAMILY_ART_SKIP"],
    "$FAMILY_CANDIDATE_HEAD": _LIMIT_FAMILY_SITES["$FAMILY_CANDIDATE_HEAD"],
    "$FAMILY_CANDIDATE_VALUES": """\
        const int q_index = candidate < 2 * dofs ? limit_q_index.data[dof] : -1;
        float qdot_max = 0.0f;
""",
    "$FAMILY_LAW": """\
        if (q_index >= 0) {
            qdot_max = wp::index(joint_velocity_limit, dof);
            // A non-positive limit is "unlimited"; a driven DOF of the fused clamp gets no rows.
            active = !(qdot_max <= 0.0f);
            if (active && skip_driven != 0) active = drive_slot.data[dof] < 0;
            // The proximity gate on the predicted velocity, read only with a positive fraction.
            if (active && activation_fraction > 0.0f)
                active = !(fabsf(v_hat.data[dof]) < activation_fraction * qdot_max);
        }
""",
    "$FAMILY_VISIT": _LIMIT_FAMILY_SITES["$FAMILY_VISIT"],
    "$FAMILY_SELECT_CUDA": """\
            const int selected_side = __shfl_sync(MASK, side, source);
            const float selected_qdot_max = __shfl_sync(MASK, qdot_max, source);
""",
    "$FAMILY_SELECT_CPU": """\
            const int selected_side = side;
            const float selected_qdot_max = qdot_max;
""",
    "$FAMILY_SIGN_DECL": _LIMIT_FAMILY_SITES["$FAMILY_SIGN_DECL"],
    "$FAMILY_VALUE": _LIMIT_FAMILY_SITES["$FAMILY_VALUE"],
    "$FAMILY_INCIDENT": _LIMIT_FAMILY_SITES["$FAMILY_INCIDENT"],
    "$FAMILY_ROW_TYPE": str(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT),
    "$FAMILY_ROW_TAIL": """\
                // Steel's velocity-limit row: rhs = -target + J v = qdot_max +/- v, negative when its side is violated.
                row_mu.data[row_offset] = 0.0f;
                phi.data[row_offset] = 0.0f;
                target_velocity.data[row_offset] = -selected_qdot_max;
""",
}


# The mimic family's sites: Steel's mimic rows (kernels.py allocate_mimic_slots and populate_mimic_J_for_size), one
# candidate per mimic row of the articulation, in its range of the mimic table (mimic_art_start, mimic_art_list: the
# rows by articulation, each articulation's in table order). A row of a disabled deprecated constraint
# (Model.constraint_mimic_enabled, read every step) builds none; joint-owned rows are always enabled. The row is
# J = e_follower - c1 e_leader on its two DOFs, so its response is the follower's factor column less c1 times the
# leader's, over the OR of their support masks (_MIMIC_MASK_SITES); phi = q_follower - c1 q_leader - c0, no target,
# and the coefficients (c0, c1) are read every step: the deprecated constraint's, or the follower joint's
# joint_mimic_coeffs. The candidate's DOF (local_dof, the template's selected_dof) is the follower's. The model's
# arrays are read through their strides.
_MIMIC_FAMILY_SITES = {
    "$FAMILY_ART_SKIP": "",
    "$FAMILY_CANDIDATE_HEAD": """\
    const int mimic_begin = mimic_art_start.data[art];
    const int mimic_rows = mimic_art_start.data[art + 1] - mimic_begin;
    for (int base = 0; base < mimic_rows; base += width) {
        const int candidate = base + lane;
        const int mimic = candidate < mimic_rows ? mimic_art_list.data[mimic_begin + candidate] : -1;
        const int local_dof = mimic >= 0 ? mimic_dof0.data[mimic] - start : 0;
""",
    "$FAMILY_CANDIDATE_VALUES": """\
        int leader_dof = 0;
        float coefficient = 0.0f;
        float violation = 0.0f;
""",
    "$FAMILY_LAW": """\
        if (mimic >= 0) {
            const int legacy = mimic_legacy.data[mimic];
            // A disabled deprecated constraint builds no row; joint-owned rows are always enabled.
            active = legacy < 0 || wp::index(mimic_enabled, legacy);
            float offset = 0.0f;
            if (legacy >= 0) {
                offset = wp::index(mimic_coef0, legacy);
                coefficient = wp::index(mimic_coef1, legacy);
            } else {
                const auto coefficients = wp::index(joint_mimic_coeffs, mimic_owner.data[mimic]);
                offset = coefficients.c[0];
                coefficient = coefficients.c[1];
            }
            leader_dof = mimic_dof1.data[mimic] - start;
            violation = joint_q.data[mimic_q0.data[mimic]] - coefficient * joint_q.data[mimic_q1.data[mimic]] - offset;
        }
""",
    "$FAMILY_VISIT": """\
        // Visit the ballot in lane order, the articulation's mimic order. A COUNT pass only advances the cursor.
""",
    "$FAMILY_SELECT_CUDA": """\
            const int selected_leader = __shfl_sync(MASK, leader_dof, source);
            const float selected_coefficient = __shfl_sync(MASK, coefficient, source);
            const float selected_violation = __shfl_sync(MASK, violation, source);
""",
    "$FAMILY_SELECT_CPU": """\
            const int selected_leader = leader_dof;
            const float selected_coefficient = coefficient;
            const float selected_violation = violation;
""",
    "$FAMILY_SIGN_DECL": """\
            const int leader_column = inverse_permutation.data[selected_leader];
""",
    "$FAMILY_VALUE": """\
                // J = e_follower - c1 e_leader: the follower's factor column less c1 times the leader's.
                const int leader_entry = factor_lookup.data[node * dofs + leader_column];
                const float follower_value = entry >= 0 ? inverse_factor.data[factor_base + entry] : 0.0f;
                const float leader_value = leader_entry >= 0 ? inverse_factor.data[factor_base + leader_entry] : 0.0f;
                const float value = follower_value - selected_coefficient * leader_value;
""",
    "$FAMILY_INCIDENT": "v_hat.data[start + selected_dof] - selected_coefficient * v_hat.data[start + selected_leader]",
    "$FAMILY_ROW_TYPE": str(PGS_CONSTRAINT_TYPE_MIMIC),
    "$FAMILY_ROW_TAIL": """\
                // Steel's mimic row: the signed violation, no friction and no target.
                row_mu.data[row_offset] = 0.0f;
                phi.data[row_offset] = selected_violation;
                target_velocity.data[row_offset] = 0.0f;
""",
}


def _expand_family(template: str, sites: dict[str, str]) -> str:
    """Fill the family sites (``$FAMILY_*``) of ``template`` with one row family's ``sites``.

    Every site of the template must be given and present; a family's text differs from another's only
    at its sites.
    """
    expected = set(re.findall(r"\$FAMILY_[A-Z_]+", template))
    if set(sites) != expected:
        raise AssertionError(f"family sites {sorted(sites)} do not match the template's {sorted(expected)}")
    # Longer placeholders first, so that none is replaced inside another.
    for placeholder in sorted(sites, key=len, reverse=True):
        template = template.replace(placeholder, sites[placeholder])
    return template


def _joint_dof_rows_snippet(
    family_sites: dict[str, str], mask_words: int, mask_sites: dict[str, tuple[str, str]] | None = None
) -> str:
    """:data:`_JOINT_DOF_ROWS_TEMPLATE` for one row family's sites and support masks of ``mask_words`` words.

    ``mask_sites`` are the mask sites to fill, :data:`_JOINT_DOF_ROWS_MASK_SITES` by default (every family but the
    mimic family's, :data:`_MIMIC_MASK_SITES`).
    """
    sites = _JOINT_DOF_ROWS_MASK_SITES if mask_sites is None else mask_sites
    return (
        _expand_masks(_expand_family(_JOINT_DOF_ROWS_TEMPLATE, family_sites), sites, mask_words)
        .replace("$ROW_SLOTS_ATOMIC", str(ROW_SLOTS_ATOMIC))
        .replace("$ROW_SLOTS_COUNT", str(ROW_SLOTS_COUNT))
    )


# The internal-row builders' mask sites (:func:`_expand_masks`): the selected DOF's support mask,
# the coordinate loop's membership test, the coordinate's rank (its entry in the row) and the
# mask's coordinates.
_JOINT_DOF_ROWS_MASK_SITES = {
    "$MASK_HELPERS": (
        """    const auto bit_count = [](unsigned long long mask) {
#if defined(__CUDA_ARCH__)
        return __popcll(mask);
#else
        int count = 0;
        while (mask != 0) { mask &= mask - 1; ++count; }
        return count;
#endif
    };
""",
        _mask_helpers_template(("mask_test", "mask_rank")),
    ),
    "$MASK_LOAD": (
        """            const unsigned long long support_mask = dof_mask.data[selected_dof];
""",
        """            unsigned long long support_mask[MASK_WORDS];
            for (int word = 0; word < MASK_WORDS; ++word)
                support_mask[word] = dof_mask.data[selected_dof * MASK_WORDS + word];
""",
    ),
    "$MASK_SKIP": (
        """                const unsigned long long bit = 1ull << node;
                if ((support_mask & bit) == 0) continue;
""",
        """                if (!mask_test(support_mask, node)) continue;
""",
    ),
    "$MASK_RANK": ("bit_count(support_mask & (bit - 1ull))", "mask_rank(support_mask, node)"),
    "$MASK_COUNT": ("bit_count(support_mask)", "mask_rank(support_mask, 64 * MASK_WORDS)"),
}

# The mimic family's mask sites: the row's support is the OR of its follower's and its leader's support masks (a
# mimic row's two DOFs need not share a chain, as two fingers of a gripper do not); the other sites are every
# family's.
_MIMIC_MASK_SITES = {
    **_JOINT_DOF_ROWS_MASK_SITES,
    "$MASK_LOAD": (
        """            const unsigned long long support_mask =
                dof_mask.data[selected_dof] | dof_mask.data[selected_leader];
""",
        """            unsigned long long support_mask[MASK_WORDS];
            for (int word = 0; word < MASK_WORDS; ++word)
                support_mask[word] = dof_mask.data[selected_dof * MASK_WORDS + word]
                    | dof_mask.data[selected_leader * MASK_WORDS + word];
""",
    ),
}


def sparse_joint_limit_rows_snippet(mask_words: int = 1) -> str:
    """The native snippet of the sparse joint-limit row builder for support masks of ``mask_words`` words.

    ``dof_mask`` then holds ``mask_words`` consecutive words per DOF. It is the joint-limit family's
    expansion of the internal-row builders' one template (:data:`_JOINT_DOF_ROWS_TEMPLATE`,
    :data:`_LIMIT_FAMILY_SITES`). With one word it is the snippet of
    :func:`~newton._src.solvers.steel.sparse_contact.build_sparse_joint_limit_rows`, Steel's own builder,
    character for character; with more, the mask lines call the helpers of :data:`_MASK_HELPERS_CUDA`.
    """
    return _joint_dof_rows_snippet(_LIMIT_FAMILY_SITES, mask_words)


@cache
def get_sparse_joint_limit_rows_kernel(mask_words: int = 1) -> wp.Kernel:
    """The sparse joint-limit row builder for support masks of ``mask_words`` ``uint64`` words.

    With one word it is Steel's builder,
    :func:`~newton._src.solvers.steel.sparse_contact.build_sparse_joint_limit_rows`, itself. With more it
    is a kernel of the same arguments and launch (:func:`sparse_joint_limit_rows_snippet`) whose
    ``dof_mask`` holds ``mask_words`` consecutive words per DOF, as
    :meth:`~newton._src.solvers.steel.sparse_mass_matrix._SparseMassMatrixPlan.to_device` uploads a
    plan's ``ancestor_mask``.
    """
    words = int(mask_words)
    if words == 1:
        return build_sparse_joint_limit_rows
    snippet = sparse_joint_limit_rows_snippet(words)

    @wp.func_native(snippet)
    def limit_rows_native(
        tid: int,
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        limit_q_index: wp.array[int],
        joint_limit_lower: wp.array[float],
        joint_limit_upper: wp.array[float],
        joint_q: wp.array[float],
        activation_gap: float,
        dof_mask: wp.array[wp.uint64],
        inverse_permutation: wp.array[int],
        factor_lookup: wp.array2d[int],
        inverse_factor: wp.array2d[float],
        v_hat: wp.array[float],
        slot_counter: wp.array[int],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        phi: wp.array2d[float],
        target_velocity: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
        row_slot_mode: int,
        element_row_cursor: wp.array[int],
    ): ...

    def limit_rows_kernel(  # pragma: no cover - compiled by wp.kernel
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        limit_q_index: wp.array[int],
        joint_limit_lower: wp.array[float],
        joint_limit_upper: wp.array[float],
        joint_q: wp.array[float],
        activation_gap: float,
        dof_mask: wp.array[wp.uint64],
        inverse_permutation: wp.array[int],
        factor_lookup: wp.array2d[int],
        inverse_factor: wp.array2d[float],
        v_hat: wp.array[float],
        slot_counter: wp.array[int],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        phi: wp.array2d[float],
        target_velocity: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
        row_slot_mode: int,
        element_row_cursor: wp.array[int],
    ):
        limit_rows_native(
            wp.tid(),
            group_to_art,
            art_to_world,
            articulation_world_dof_offset,
            articulation_dof_start,
            limit_q_index,
            joint_limit_lower,
            joint_limit_upper,
            joint_q,
            activation_gap,
            dof_mask,
            inverse_permutation,
            factor_lookup,
            inverse_factor,
            v_hat,
            slot_counter,
            row_type,
            row_parent,
            row_mu,
            phi,
            target_velocity,
            row_dof,
            row_factor,
            row_incident,
            diagonal,
            row_slot_mode,
            element_row_cursor,
        )

    limit_rows_kernel.__name__ = f"build_sparse_joint_limit_rows_w{words}"
    limit_rows_kernel.__qualname__ = limit_rows_kernel.__name__
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(limit_rows_kernel)


# --- The internal unit's row families (scene_solve) ----------------------------------------------

FAMILY_DRIVE = 0
FAMILY_MIMIC = 1
FAMILY_CONNECT = 2
FAMILY_LIMIT = 3
FAMILY_JOINT_FRICTION = 4
FAMILY_VELOCITY_LIMIT = 5
FAMILY_COUNT = 6
"""The per-articulation row families of the scene, in Steel's slot order.

Steel lays out a world's dense rows as ``[drives][mimic][connect][joint limits][joint friction][joint velocity
limits]`` before its contacts (``_stage4_build_rows``). The scene's internal unit holds one articulation's rows of the
first five families (``FAMILY_DRIVE`` to ``FAMILY_JOINT_FRICTION``) in that order; the velocity-limit family's rows
are late rows, which an articulation's *late unit* holds after every other unit of the row order. The drive family's
builder is :func:`get_sparse_drive_rows_kernel`, the mimic family's :func:`get_sparse_mimic_rows_kernel`, the connect
family's :func:`get_scene_connect_rows_kernel` (through the contact response, one launch for every loop closure), the
joint-limit family's :func:`get_sparse_joint_limit_rows_kernel`, the joint-friction family's
:func:`get_sparse_joint_friction_rows_kernel` and the velocity-limit family's
:func:`get_sparse_joint_velocity_limit_rows_kernel`.
"""


def sparse_joint_friction_rows_snippet(mask_words: int = 1) -> str:
    """The native snippet of :func:`get_sparse_joint_friction_rows_kernel` for support masks of ``mask_words`` words.

    It is the joint-friction family's expansion of the internal-row builders' one template
    (:data:`_JOINT_DOF_ROWS_TEMPLATE`, :data:`_FRICTION_FAMILY_SITES`): the joint-limit builder's
    reservation, ballot visit and sparse response, with one candidate per DOF, Steel's joint friction
    law and ``J = 1``; it differs from :func:`sparse_joint_limit_rows_snippet` only at the family sites.
    """
    return _joint_dof_rows_snippet(_FRICTION_FAMILY_SITES, mask_words)


@cache
def get_sparse_joint_friction_rows_kernel(mask_words: int = 1) -> wp.Kernel:
    """The sparse joint-friction row builder for support masks of ``mask_words`` ``uint64`` words.

    Steel's joint friction law (``solver_steel._get_joint_friction_warp_kernel``) on
    the factor-coordinate rows: one warp per articulation of the group, ``group_to_art``, skipping an
    articulation whose ``articulation_rows_active`` entry is ``0``. A DOF whose joint type supports friction
    (``friction_dof_supported``), that is not kinematic (``kinematic_dof_mask``) and whose
    :attr:`~newton.Model.joint_friction` ``f`` (read through its stride) gives ``f dt > 0`` with ``f``
    finite builds one row: ``J = 1`` on the DOF, type ``PGS_CONSTRAINT_TYPE_JOINT_FRICTION``, no parent,
    ``phi = 0``, no target, ``row_mu`` and ``row_bound`` the bound ``f dt``, and the response, diagonal and
    incident velocity of :func:`get_sparse_joint_limit_rows_kernel`'s rows with the sign ``+1``.

    The modes are the joint-limit builder's: ``ROW_SLOTS_COUNT`` adds each articulation's rows to
    ``element_row_cursor`` and writes none; ``ROW_SLOTS_WRITE`` writes them from ``element_row_cursor[art]``
    in DOF order, advancing it, and none at or past the row capacity; ``ROW_SLOTS_ATOMIC`` reserves them
    from ``slot_counter[world]``. Launch ``32 * group_count`` threads in blocks divisible by 32.
    """
    words = int(mask_words)
    if words < 1:
        raise ValueError(f"mask_words must be at least 1, got {words}")
    snippet = sparse_joint_friction_rows_snippet(words)

    @wp.func_native(snippet)
    def friction_rows_native(
        tid: int,
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        articulation_rows_active: wp.array[int],
        friction_dof_supported: wp.array[int],
        kinematic_dof_mask: wp.array[int],
        joint_friction: wp.array[float],
        dt: float,
        dof_mask: wp.array[wp.uint64],
        inverse_permutation: wp.array[int],
        factor_lookup: wp.array2d[int],
        inverse_factor: wp.array2d[float],
        v_hat: wp.array[float],
        slot_counter: wp.array[int],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        phi: wp.array2d[float],
        target_velocity: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
        row_bound: wp.array2d[float],
        row_slot_mode: int,
        element_row_cursor: wp.array[int],
    ): ...

    def friction_rows_kernel(  # pragma: no cover - compiled by wp.kernel
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        articulation_rows_active: wp.array[int],
        friction_dof_supported: wp.array[int],
        kinematic_dof_mask: wp.array[int],
        joint_friction: wp.array[float],
        dt: float,
        dof_mask: wp.array[wp.uint64],
        inverse_permutation: wp.array[int],
        factor_lookup: wp.array2d[int],
        inverse_factor: wp.array2d[float],
        v_hat: wp.array[float],
        slot_counter: wp.array[int],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        phi: wp.array2d[float],
        target_velocity: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
        row_bound: wp.array2d[float],
        row_slot_mode: int,
        element_row_cursor: wp.array[int],
    ):
        friction_rows_native(
            wp.tid(),
            group_to_art,
            art_to_world,
            articulation_world_dof_offset,
            articulation_dof_start,
            articulation_rows_active,
            friction_dof_supported,
            kinematic_dof_mask,
            joint_friction,
            dt,
            dof_mask,
            inverse_permutation,
            factor_lookup,
            inverse_factor,
            v_hat,
            slot_counter,
            row_type,
            row_parent,
            row_mu,
            phi,
            target_velocity,
            row_dof,
            row_factor,
            row_incident,
            diagonal,
            row_bound,
            row_slot_mode,
            element_row_cursor,
        )

    friction_rows_kernel.__name__ = f"build_sparse_joint_friction_rows_w{words}"
    friction_rows_kernel.__qualname__ = friction_rows_kernel.__name__
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(friction_rows_kernel)


def sparse_drive_rows_snippet(mask_words: int = 1, *, fused: bool = False) -> str:
    """The native snippet of :func:`get_sparse_drive_rows_kernel` for support masks of ``mask_words`` words.

    It is the drive family's expansion of the internal-row builders' one template
    (:data:`_JOINT_DOF_ROWS_TEMPLATE`, :data:`_DRIVE_FAMILY_SITES`): the joint-limit builder's reservation,
    ballot visit and sparse response, with one candidate per DOF, Steel's drive law and ``J = 1``; it differs
    from the other families' snippets only at the family sites. ``fused`` (the fused velocity clamp) expands
    :data:`_FUSED_DRIVE_FAMILY_SITES`, whose row tail also writes ``drive_vel_limit``; without it the text is the
    unfused drive family's, unchanged.
    """
    return _joint_dof_rows_snippet(_FUSED_DRIVE_FAMILY_SITES if fused else _DRIVE_FAMILY_SITES, mask_words)


@cache
def get_sparse_drive_rows_kernel(mask_words: int = 1, *, fused: bool = False) -> wp.Kernel:
    """The sparse drive row builder (``drive_mode="rows"``) for support masks of ``mask_words`` ``uint64`` words.

    Steel's drive rows (``kernels.allocate_physx_drive_slots`` and
    ``kernels.populate_physx_drive_J_for_size``) on the factor-coordinate rows: one warp per articulation of the
    group, ``group_to_art``. A DOF of a PRISMATIC, REVOLUTE or D6 joint (``limit_q_index >= 0``, its
    coordinate) builds one row unless its :attr:`~newton.Model.joint_target_ke` and
    :attr:`~newton.Model.joint_target_kd` are both non-positive: ``J = 1`` on the DOF, type
    ``PGS_CONSTRAINT_TYPE_JOINT_TARGET``, no parent, ``mu = phi = 0``, the target velocity
    ``joint_target_qd[dof]``, and the drive parameters ``drive_stiffness`` (``ke``), ``drive_damping``
    (``kd``), ``drive_geom_error`` (``joint_target_q[q] - joint_q[q]`` at the DOF's coordinate ``q``) and
    ``drive_max_force`` (:attr:`~newton.Model.joint_effort_limit`), from which
    :func:`~newton._src.solvers.steel.scene_solve.scene_drive_descriptors` computes the row's force-drive
    update once its diagonal is final. The response, diagonal and incident velocity are those of
    :func:`get_sparse_joint_limit_rows_kernel`'s rows with the sign ``+1``.

    With ``fused`` (``fuse_joint_velocity_limits`` with joint velocity limits) each row also records its DOF's
    :attr:`~newton.Model.joint_velocity_limit` in ``drive_vel_limit``, ``inf`` when it is non-positive or
    non-finite, the limit of phase C's fused clamp (Steel's ``fuse_vel_limits``,
    ``kernels.populate_physx_drive_J_for_size``); without it neither ``joint_velocity_limit`` nor ``drive_vel_limit`` is
    read or written. Both builds take both arguments.

    The modes are the joint-limit builder's: ``ROW_SLOTS_COUNT`` adds each articulation's rows to
    ``element_row_cursor`` and writes none; ``ROW_SLOTS_WRITE`` writes them from ``element_row_cursor[art]``
    in DOF order, advancing it, and none at or past the row capacity; ``ROW_SLOTS_ATOMIC`` reserves them
    from ``slot_counter[world]``. Launch ``32 * group_count`` threads in blocks divisible by 32.
    """
    words = int(mask_words)
    if words < 1:
        raise ValueError(f"mask_words must be at least 1, got {words}")
    snippet = sparse_drive_rows_snippet(words, fused=fused)

    @wp.func_native(snippet)
    def drive_rows_native(
        tid: int,
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        limit_q_index: wp.array[int],
        joint_target_ke: wp.array[float],
        joint_target_kd: wp.array[float],
        joint_effort_limit: wp.array[float],
        joint_q: wp.array[float],
        joint_target_q: wp.array[float],
        joint_target_qd: wp.array[float],
        joint_velocity_limit: wp.array[float],
        dof_mask: wp.array[wp.uint64],
        inverse_permutation: wp.array[int],
        factor_lookup: wp.array2d[int],
        inverse_factor: wp.array2d[float],
        v_hat: wp.array[float],
        slot_counter: wp.array[int],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        phi: wp.array2d[float],
        target_velocity: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
        drive_stiffness: wp.array2d[float],
        drive_damping: wp.array2d[float],
        drive_geom_error: wp.array2d[float],
        drive_max_force: wp.array2d[float],
        drive_vel_limit: wp.array2d[float],
        row_slot_mode: int,
        element_row_cursor: wp.array[int],
    ): ...

    def drive_rows_kernel(  # pragma: no cover - compiled by wp.kernel
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        limit_q_index: wp.array[int],
        joint_target_ke: wp.array[float],
        joint_target_kd: wp.array[float],
        joint_effort_limit: wp.array[float],
        joint_q: wp.array[float],
        joint_target_q: wp.array[float],
        joint_target_qd: wp.array[float],
        joint_velocity_limit: wp.array[float],
        dof_mask: wp.array[wp.uint64],
        inverse_permutation: wp.array[int],
        factor_lookup: wp.array2d[int],
        inverse_factor: wp.array2d[float],
        v_hat: wp.array[float],
        slot_counter: wp.array[int],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        phi: wp.array2d[float],
        target_velocity: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
        drive_stiffness: wp.array2d[float],
        drive_damping: wp.array2d[float],
        drive_geom_error: wp.array2d[float],
        drive_max_force: wp.array2d[float],
        drive_vel_limit: wp.array2d[float],
        row_slot_mode: int,
        element_row_cursor: wp.array[int],
    ):
        drive_rows_native(
            wp.tid(),
            group_to_art,
            art_to_world,
            articulation_world_dof_offset,
            articulation_dof_start,
            limit_q_index,
            joint_target_ke,
            joint_target_kd,
            joint_effort_limit,
            joint_q,
            joint_target_q,
            joint_target_qd,
            joint_velocity_limit,
            dof_mask,
            inverse_permutation,
            factor_lookup,
            inverse_factor,
            v_hat,
            slot_counter,
            row_type,
            row_parent,
            row_mu,
            phi,
            target_velocity,
            row_dof,
            row_factor,
            row_incident,
            diagonal,
            drive_stiffness,
            drive_damping,
            drive_geom_error,
            drive_max_force,
            drive_vel_limit,
            row_slot_mode,
            element_row_cursor,
        )

    drive_rows_kernel.__name__ = f"build_sparse_drive_rows{'_fused' if fused else ''}_w{words}"
    drive_rows_kernel.__qualname__ = drive_rows_kernel.__name__
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(drive_rows_kernel)


def sparse_joint_velocity_limit_rows_snippet(mask_words: int = 1) -> str:
    """The native snippet of :func:`get_sparse_joint_velocity_limit_rows_kernel` for ``mask_words``-word masks.

    It is the velocity-limit family's expansion of the internal-row builders' one template
    (:data:`_JOINT_DOF_ROWS_TEMPLATE`, :data:`_VELOCITY_LIMIT_FAMILY_SITES`): the joint-limit builder's
    reservation, ballot visit and sparse response, with its two candidates per DOF and signs, Steel's
    velocity-limit law and the sleeping skip of the joint-friction family; it differs from the other families'
    snippets only at the family sites.
    """
    return _joint_dof_rows_snippet(_VELOCITY_LIMIT_FAMILY_SITES, mask_words)


@cache
def get_sparse_joint_velocity_limit_rows_kernel(mask_words: int = 1) -> wp.Kernel:
    """The sparse joint velocity-limit row builder (``enable_joint_velocity_limits``) for ``mask_words``-word masks.

    Steel's velocity-limit rows (``kernels.allocate_joint_velocity_limit_slots`` and
    ``populate_joint_velocity_limit_J_for_size``) on the factor-coordinate rows: one warp per articulation of the
    group, ``group_to_art``, skipping an articulation whose ``articulation_rows_active`` entry is ``0`` (sleeping
    articulations reserve none). A DOF of a PRISMATIC, REVOLUTE or D6 joint (``limit_q_index >= 0``) with a
    :attr:`~newton.Model.joint_velocity_limit` ``qdot_max`` that is not non-positive builds two rows, the lower
    bound's (``J = +1`` on the DOF) then the upper's (``J = -1``), typed
    ``PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT``, no parent, ``mu = phi = 0`` and the target velocity
    ``-qdot_max`` for both, unless ``skip_driven`` (``fuse_joint_velocity_limits`` in effect) and
    ``drive_slot[dof] >= 0`` (the fused clamp limits the DOF), or ``activation_fraction > 0`` and
    ``|v_hat[dof]| < activation_fraction * qdot_max`` (the proximity gate on the predicted velocity, which a
    fraction of ``0`` does not read). The response, diagonal and incident velocity are those of
    :func:`get_sparse_joint_limit_rows_kernel`'s rows with the same signs.

    The modes are the joint-limit builder's: ``ROW_SLOTS_COUNT`` adds each articulation's rows to
    ``element_row_cursor`` and writes none; ``ROW_SLOTS_WRITE`` writes them from ``element_row_cursor[art]``
    (the articulation's late unit's first row) in Steel's slot order, advancing it, and none at or past the row
    capacity; ``ROW_SLOTS_ATOMIC`` reserves them from ``slot_counter[world]``. Launch ``32 * group_count`` threads
    in blocks divisible by 32.
    """
    words = int(mask_words)
    if words < 1:
        raise ValueError(f"mask_words must be at least 1, got {words}")
    snippet = sparse_joint_velocity_limit_rows_snippet(words)

    @wp.func_native(snippet)
    def velocity_limit_rows_native(
        tid: int,
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        articulation_rows_active: wp.array[int],
        limit_q_index: wp.array[int],
        joint_velocity_limit: wp.array[float],
        activation_fraction: float,
        drive_slot: wp.array[int],
        skip_driven: int,
        dof_mask: wp.array[wp.uint64],
        inverse_permutation: wp.array[int],
        factor_lookup: wp.array2d[int],
        inverse_factor: wp.array2d[float],
        v_hat: wp.array[float],
        slot_counter: wp.array[int],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        phi: wp.array2d[float],
        target_velocity: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
        row_slot_mode: int,
        element_row_cursor: wp.array[int],
    ): ...

    def velocity_limit_rows_kernel(  # pragma: no cover - compiled by wp.kernel
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        articulation_rows_active: wp.array[int],
        limit_q_index: wp.array[int],
        joint_velocity_limit: wp.array[float],
        activation_fraction: float,
        drive_slot: wp.array[int],
        skip_driven: int,
        dof_mask: wp.array[wp.uint64],
        inverse_permutation: wp.array[int],
        factor_lookup: wp.array2d[int],
        inverse_factor: wp.array2d[float],
        v_hat: wp.array[float],
        slot_counter: wp.array[int],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        phi: wp.array2d[float],
        target_velocity: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
        row_slot_mode: int,
        element_row_cursor: wp.array[int],
    ):
        velocity_limit_rows_native(
            wp.tid(),
            group_to_art,
            art_to_world,
            articulation_world_dof_offset,
            articulation_dof_start,
            articulation_rows_active,
            limit_q_index,
            joint_velocity_limit,
            activation_fraction,
            drive_slot,
            skip_driven,
            dof_mask,
            inverse_permutation,
            factor_lookup,
            inverse_factor,
            v_hat,
            slot_counter,
            row_type,
            row_parent,
            row_mu,
            phi,
            target_velocity,
            row_dof,
            row_factor,
            row_incident,
            diagonal,
            row_slot_mode,
            element_row_cursor,
        )

    velocity_limit_rows_kernel.__name__ = f"build_sparse_joint_velocity_limit_rows_w{words}"
    velocity_limit_rows_kernel.__qualname__ = velocity_limit_rows_kernel.__name__
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(
        velocity_limit_rows_kernel
    )


# A free rigid body's velocity-limit rows: Steel's free-body velocity limits
# (kernels.allocate_rigid_velocity_limit_slots and kernels.populate_rigid_velocity_limit_rows), twelve candidates per
# body, the lower (J = +e_axis) then the upper (J = -e_axis) bound of each of its six root DOFs [lin_x, lin_y, lin_z,
# ang_x, ang_y, ang_z], in Steel's slot order. They are the body's late unit, as an articulation's joint
# velocity-limit rows are its.
FREE_VELOCITY_LIMIT_CANDIDATES = wp.constant(12)
"""Candidates of a free body's velocity-limit rows: two (lower, upper) per root DOF, Steel's twelve slots."""


@wp.func
def free_velocity_limit_candidate(
    candidate: int,
    body: int,
    art: int,
    rigid_body_max_linear_velocity: wp.array[float],
    rigid_body_max_angular_velocity: wp.array[float],
    articulation_root_dof_start: wp.array[int],
    free_root_anchor: wp.array[wp.quat],
    activation_fraction: float,
    v_hat: wp.array[float],
) -> float:
    """The limit of a free body's velocity-limit candidate when it builds a row, else ``0``: Steel's per-axis test.

    Candidate ``2 * axis + side`` bounds root DOF ``axis`` (``kernels.allocate_rigid_velocity_limit_slots``): the
    body's ``rigid_body_max_linear_velocity`` for the linear axes and ``rigid_body_max_angular_velocity`` for the
    angular ones, skipped when it is non-positive or not finite; with a positive ``activation_fraction`` the pair is
    built only when the axis's world-axis speed ``|v|`` satisfies ``|v| >= fraction * limit``, read from the
    predicted velocity ``v_hat`` at the root DOFs and rotated by the FREE joint's parent anchor into the world (the
    limits bound world-axis components, ``v_hat`` is in the anchor frame); a fraction of ``0`` reads no velocity. Both
    sides of an axis share its test. The one predicate of every mode of
    :func:`build_sparse_free_velocity_limit_rows`.
    """
    axis = candidate // 2
    limit = rigid_body_max_linear_velocity[body]
    if axis >= 3:
        limit = rigid_body_max_angular_velocity[body]
    if limit <= 0.0 or not wp.isfinite(limit):
        return 0.0
    if activation_fraction > 0.0:
        # The limits bound world-axis components; v_hat is in the parent-anchor frame.
        root_dof = articulation_root_dof_start[art] + 3 * (axis // 3)
        local = wp.vec3(v_hat[root_dof], v_hat[root_dof + 1], v_hat[root_dof + 2])
        speed = _anchor_to_world(free_root_anchor[art], local)[axis % 3]
        if wp.abs(speed) < activation_fraction * limit:
            return 0.0
    return limit


@wp.kernel(enable_backward=False)
def build_sparse_free_velocity_limit_rows(
    free_rigid_body_indices: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    is_free_rigid: wp.array[int],
    body_flags: wp.array[wp.int32],
    articulation_rows_active: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_dof_start: wp.array[int],
    articulation_root_dof_start: wp.array[int],
    art_group_idx: wp.array[int],
    rigid_body_max_linear_velocity: wp.array[float],
    rigid_body_max_angular_velocity: wp.array[float],
    free_root_anchor: wp.array[wp.quat],
    activation_fraction: float,
    free_inverse_factor: wp.array3d[float],
    v_hat: wp.array[float],
    # outputs
    slot_counter: wp.array[int],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    phi: wp.array2d[float],
    target_velocity: wp.array2d[float],
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    row_free_response: wp.array3d[float],
    row_incident: wp.array2d[float],
    diagonal: wp.array2d[float],
    row_slot_mode: int,
    element_row_cursor: wp.array[int],
):
    """Build a free rigid body's velocity-limit rows, the late unit of its side (grid: free bodies).

    Steel's ``kernels.allocate_rigid_velocity_limit_slots`` and ``kernels.populate_rigid_velocity_limit_rows``,
    one thread per body of ``free_rigid_body_indices``, on the scene's rows: a body that is not a free
    rigid body's, kinematic, asleep (``articulation_rows_active``) or without a solve world builds none
    (as Steel's allocation); each candidate that :func:`free_velocity_limit_candidate` passes builds a row in
    Steel's slot order, typed ``PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT``, no parent, ``mu = phi = 0`` and the
    target velocity ``-limit``, so that its bias is the limit Steel keeps in ``mf_phi``
    (``populate_rigid_velocity_limit_rows``, and ``compute_mf_effective_mass_and_rhs``).
    Its Jacobian is ``sign * e_axis`` on the world axis (lower ``+1``, upper ``-1``), expressed in the FREE joint's
    parent-anchor frame of ``joint_qd`` as Steel's ``rotate_mf_rows_to_free_root_anchor`` expresses it
    (both blocks through ``quat_rotate_inv`` unless the anchor is the identity, ``_rotate_row_to_anchor``): its
    six entries are the body's physical coordinates ``articulation_world_dof_offset[art] + 0..5``, the row's
    free-body prefix. Its
    response ``H^-1 J`` (``row_free_response``), diagonal ``J H^-1 J`` and incident velocity ``J v_hat`` follow the
    free-body branch of the contact response, through the body's 6 x 6 ``L^-1`` in ``free_inverse_factor`` at
    ``art_group_idx[art]`` (Steel's size-6 group).

    The modes are the late unit's cursor contract (:func:`get_sparse_joint_velocity_limit_rows_kernel`), through
    Steel's ``reserve_world_rows`` with the articulation as the element: ``ROW_SLOTS_COUNT`` adds the body's rows to
    ``element_row_cursor[art]`` and writes none; ``ROW_SLOTS_WRITE`` writes them from ``element_row_cursor[art]``,
    advancing it, and none at or past the row capacity; ``ROW_SLOTS_ATOMIC`` reserves them from
    ``slot_counter[world]``. One predicate decides the candidates of every mode.
    """
    body = free_rigid_body_indices[wp.tid()]
    art = body_to_articulation[body]
    if art < 0:
        return
    if is_free_rigid[art] == 0:
        return
    if (body_flags[body] & BodyFlags.KINEMATIC) != 0:
        return
    if articulation_rows_active[art] == 0:
        return
    world = art_to_world[art]
    if world < 0:
        return

    active = int(0)
    count = int(0)
    for candidate in range(FREE_VELOCITY_LIMIT_CANDIDATES):
        limit = free_velocity_limit_candidate(
            candidate,
            body,
            art,
            rigid_body_max_linear_velocity,
            rigid_body_max_angular_velocity,
            articulation_root_dof_start,
            free_root_anchor,
            activation_fraction,
            v_hat,
        )
        if limit > 0.0:
            active = active | (1 << candidate)
            count += 1
    if count == 0:
        return
    first = reserve_world_rows(row_slot_mode, slot_counter, world, count, element_row_cursor, art)
    if row_slot_mode == ROW_SLOTS_COUNT:
        return

    capacity = row_type.shape[1]
    anchor = free_root_anchor[art]
    group = art_group_idx[art]
    start = articulation_dof_start[art]
    offset = articulation_world_dof_offset[art]
    rank = int(0)
    for candidate in range(FREE_VELOCITY_LIMIT_CANDIDATES):
        if ((active >> candidate) & 1) == 0:
            continue
        row = first + rank
        rank += 1
        # The reservation above still counts every over-capacity row.
        if row >= capacity:
            break
        axis = candidate // 2
        sign = float(1.0)
        if (candidate & 1) != 0:
            sign = -1.0
        limit = rigid_body_max_linear_velocity[body]
        if axis >= 3:
            limit = rigid_body_max_angular_velocity[body]
        # J = sign * e_axis in world axes, then in the anchor frame of joint_qd.
        linear = wp.vec3(0.0)
        angular = wp.vec3(0.0)
        if axis < 3:
            linear[axis] = sign
        else:
            angular[axis - 3] = sign
        if not _is_identity_rotation(anchor):
            linear = wp.quat_rotate_inv(anchor, linear)
            angular = wp.quat_rotate_inv(anchor, angular)
        jacobian = wp.spatial_vector(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])
        # The response H^-1 J = L^-T (L^-1 J), as the contact response's free-body branch forms it.
        free_factor = wp.spatial_vector()
        for node in range(6):
            value = float(0.0)
            for column in range(node + 1):
                value += free_inverse_factor[group, node, column] * jacobian[column]
            free_factor[node] = value
        norm = float(0.0)
        incident = float(0.0)
        for node in range(6):
            response = float(0.0)
            for column in range(node, 6):
                response += free_inverse_factor[group, column, node] * free_factor[column]
            value = jacobian[node]
            row_dof[world, row, node] = offset + node
            row_factor[world, row, node] = value
            row_free_response[world, row, node] = response
            norm += value * response
            incident += value * v_hat[start + node]
        for index in range(6, row_dof.shape[2]):
            row_dof[world, row, index] = -1
            row_factor[world, row, index] = 0.0
        row_incident[world, row] = incident
        diagonal[world, row] = norm
        row_type[world, row] = PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
        row_parent[world, row] = -1
        row_mu[world, row] = 0.0
        # Steel keeps the limit in mf_phi and biases the row by it; the scene's velocity-limit rows hold -limit as
        # their target velocity, whose negation prepare_scene_rows makes the bias.
        phi[world, row] = 0.0
        target_velocity[world, row] = -limit


def get_sparse_free_velocity_limit_rows_kernel() -> wp.Kernel:
    """The free-body velocity-limit row builder (:func:`build_sparse_free_velocity_limit_rows`), the late-unit
    builder of a free-rigid side (``FAMILY_VELOCITY_LIMIT``), as
    :func:`get_sparse_joint_velocity_limit_rows_kernel` is an articulation's."""
    return build_sparse_free_velocity_limit_rows


def sparse_drive_marker_snippet() -> str:
    """The native snippet of :func:`get_sparse_drive_marker_kernel`: the drive family's candidate test, one DOF each.

    The candidate lines are the drive family's own sites (:data:`_DRIVE_FAMILY_SITES`, ``$FAMILY_CANDIDATE_VALUES``
    and ``$FAMILY_LAW`` around the template's ``int active = 0;``), so the marker and the drive rows cannot drift.
    """
    candidate = (
        _DRIVE_FAMILY_SITES["$FAMILY_CANDIDATE_VALUES"]
        + "        int active = 0;\n"
        + _DRIVE_FAMILY_SITES["$FAMILY_LAW"]
    )
    return (
        """
    // One thread per DOF of the group's articulations: the DOFs the drive family's candidates visit.
    const int dofs = group_dofs;
    const int group = tid / dofs;
    if (group >= group_to_art.shape[0]) return;
    const int local_dof = tid - group * dofs;
    const int art = group_to_art.data[group];
    const int dof = articulation_dof_start.data[art] + local_dof;
"""
        + candidate
        + """        // A marker, not a row index: 0 for a DOF the drive family builds a row for.
        drive_slot.data[dof] = active != 0 ? 0 : -1;
"""
    )


@cache
def get_sparse_drive_marker_kernel() -> wp.Kernel:
    """Mark the DOFs of a factor group that the drive family builds a row for (``drive_mode="rows"``).

    Steel writes ``drive_slot[dof]``, each driven DOF's dense row, in ``kernels.allocate_physx_drive_slots`` before stage 1,
    and its ``prescale_joint_velocity_limits`` and ``allocate_joint_velocity_limit_slots`` read only
    ``drive_slot[dof] >= 0`` under the fused velocity clamp, to exclude the driven DOFs. The scene places its rows
    later, by scans, so it marks them instead: one thread per DOF ``articulation_dof_start[art] + local_dof`` of the
    group's articulations (``local_dof < group_dofs``, the drive family's candidates) writes ``0`` when the drive
    family's candidate test passes (:func:`sparse_drive_marker_snippet`: a DOF of a PRISMATIC, REVOLUTE or D6 joint
    whose target stiffness or damping is positive) and ``-1`` otherwise. The ``0`` is a marker, not a row index. DOFs
    outside the factor groups are left as they are (the caller fills ``-1`` first). Launch ``group_count * group_dofs``
    threads.
    """

    @wp.func_native(sparse_drive_marker_snippet())
    def drive_marker_native(
        tid: int,
        group_to_art: wp.array[int],
        articulation_dof_start: wp.array[int],
        group_dofs: int,
        limit_q_index: wp.array[int],
        joint_target_ke: wp.array[float],
        joint_target_kd: wp.array[float],
        drive_slot: wp.array[int],
    ): ...

    def drive_marker_kernel(  # pragma: no cover - compiled by wp.kernel
        group_to_art: wp.array[int],
        articulation_dof_start: wp.array[int],
        group_dofs: int,
        limit_q_index: wp.array[int],
        joint_target_ke: wp.array[float],
        joint_target_kd: wp.array[float],
        drive_slot: wp.array[int],
    ):
        drive_marker_native(
            wp.tid(),
            group_to_art,
            articulation_dof_start,
            group_dofs,
            limit_q_index,
            joint_target_ke,
            joint_target_kd,
            drive_slot,
        )

    drive_marker_kernel.__name__ = "mark_sparse_drive_dofs"
    drive_marker_kernel.__qualname__ = drive_marker_kernel.__name__
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(drive_marker_kernel)


def sparse_mimic_rows_snippet(mask_words: int = 1) -> str:
    """The native snippet of :func:`get_sparse_mimic_rows_kernel` for support masks of ``mask_words`` words.

    It is the mimic family's expansion of the internal-row builders' one template (:data:`_JOINT_DOF_ROWS_TEMPLATE`,
    :data:`_MIMIC_FAMILY_SITES`): the joint-limit builder's reservation, ballot visit and sparse response, with one
    candidate per mimic row, Steel's mimic law and ``J = e_follower - c1 e_leader``; it differs from the other
    families' snippets at the family sites and at the support mask's load, the OR of the row's two DOFs' masks
    (:data:`_MIMIC_MASK_SITES`).
    """
    return _joint_dof_rows_snippet(_MIMIC_FAMILY_SITES, mask_words, _MIMIC_MASK_SITES)


@cache
def get_sparse_mimic_rows_kernel(mask_words: int = 1) -> wp.Kernel:
    """The sparse mimic row builder for support masks of ``mask_words`` ``uint64`` words.

    Steel's mimic rows (``kernels.py`` ``allocate_mimic_slots`` and ``populate_mimic_J_for_size``) on the
    factor-coordinate rows: one warp per articulation of the group, ``group_to_art``, over the articulation's range
    of the solver's mimic table (``mimic_art_start``, ``mimic_art_list``; per row its follower and leader DOFs
    ``mimic_dof0`` and ``mimic_dof1``, their coordinates ``mimic_q0`` and ``mimic_q1``, its deprecated constraint
    ``mimic_legacy`` or ``-1``, and its follower joint ``mimic_owner``). A row of a deprecated constraint whose
    ``mimic_enabled`` (:attr:`~newton.Model.constraint_mimic_enabled`) is false builds none; every other row builds
    one: ``J = e_follower - c1 e_leader``, type ``PGS_CONSTRAINT_TYPE_MIMIC``, no parent, ``mu = 0``, ``phi =
    q_follower - c1 q_leader - c0`` and no target, with ``(c0, c1)`` the constraint's ``mimic_coef0`` and
    ``mimic_coef1`` or the follower joint's ``joint_mimic_coeffs``, read every step through their strides. Its
    response is the follower's factor column less ``c1`` times the leader's over the OR of their support masks, its
    diagonal the response's squared norm and its incident velocity ``J v_hat``.

    The modes are the joint-limit builder's: ``ROW_SLOTS_COUNT`` adds each articulation's rows to
    ``element_row_cursor`` and writes none; ``ROW_SLOTS_WRITE`` writes them from ``element_row_cursor[art]`` in the
    table's order, advancing it, and none at or past the row capacity; ``ROW_SLOTS_ATOMIC`` reserves them from
    ``slot_counter[world]``. Launch ``32 * group_count`` threads in blocks divisible by 32.
    """
    words = int(mask_words)
    if words < 1:
        raise ValueError(f"mask_words must be at least 1, got {words}")
    snippet = sparse_mimic_rows_snippet(words)

    @wp.func_native(snippet)
    def mimic_rows_native(
        tid: int,
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        mimic_art_start: wp.array[int],
        mimic_art_list: wp.array[int],
        mimic_dof0: wp.array[int],
        mimic_dof1: wp.array[int],
        mimic_q0: wp.array[int],
        mimic_q1: wp.array[int],
        mimic_legacy: wp.array[int],
        mimic_owner: wp.array[int],
        mimic_enabled: wp.array[wp.bool],
        mimic_coef0: wp.array[float],
        mimic_coef1: wp.array[float],
        joint_mimic_coeffs: wp.array[wp.vec2],
        joint_q: wp.array[float],
        dof_mask: wp.array[wp.uint64],
        inverse_permutation: wp.array[int],
        factor_lookup: wp.array2d[int],
        inverse_factor: wp.array2d[float],
        v_hat: wp.array[float],
        slot_counter: wp.array[int],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        phi: wp.array2d[float],
        target_velocity: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
        row_slot_mode: int,
        element_row_cursor: wp.array[int],
    ): ...

    def mimic_rows_kernel(  # pragma: no cover - compiled by wp.kernel
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        mimic_art_start: wp.array[int],
        mimic_art_list: wp.array[int],
        mimic_dof0: wp.array[int],
        mimic_dof1: wp.array[int],
        mimic_q0: wp.array[int],
        mimic_q1: wp.array[int],
        mimic_legacy: wp.array[int],
        mimic_owner: wp.array[int],
        mimic_enabled: wp.array[wp.bool],
        mimic_coef0: wp.array[float],
        mimic_coef1: wp.array[float],
        joint_mimic_coeffs: wp.array[wp.vec2],
        joint_q: wp.array[float],
        dof_mask: wp.array[wp.uint64],
        inverse_permutation: wp.array[int],
        factor_lookup: wp.array2d[int],
        inverse_factor: wp.array2d[float],
        v_hat: wp.array[float],
        slot_counter: wp.array[int],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        phi: wp.array2d[float],
        target_velocity: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
        row_slot_mode: int,
        element_row_cursor: wp.array[int],
    ):
        mimic_rows_native(
            wp.tid(),
            group_to_art,
            art_to_world,
            articulation_world_dof_offset,
            articulation_dof_start,
            mimic_art_start,
            mimic_art_list,
            mimic_dof0,
            mimic_dof1,
            mimic_q0,
            mimic_q1,
            mimic_legacy,
            mimic_owner,
            mimic_enabled,
            mimic_coef0,
            mimic_coef1,
            joint_mimic_coeffs,
            joint_q,
            dof_mask,
            inverse_permutation,
            factor_lookup,
            inverse_factor,
            v_hat,
            slot_counter,
            row_type,
            row_parent,
            row_mu,
            phi,
            target_velocity,
            row_dof,
            row_factor,
            row_incident,
            diagonal,
            row_slot_mode,
            element_row_cursor,
        )

    mimic_rows_kernel.__name__ = f"build_sparse_mimic_rows_w{words}"
    mimic_rows_kernel.__qualname__ = mimic_rows_kernel.__name__
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(mimic_rows_kernel)


def get_sparse_joint_dof_rows_kernel(family: int, mask_words: int = 1, *, fused: bool = False) -> wp.Kernel:
    """The sparse row builder of one per-articulation row family (:data:`FAMILY_COUNT`) for ``mask_words``-word masks.

    Every builder runs one warp per articulation of a factor group in the COUNT and WRITE modes of
    :func:`build_sparse_joint_limit_rows` (and its atomic mode). ``FAMILY_DRIVE`` is
    :func:`get_sparse_drive_rows_kernel`, its fused velocity clamp's build with ``fused``; ``FAMILY_MIMIC`` is
    :func:`get_sparse_mimic_rows_kernel`; ``FAMILY_LIMIT`` is
    :func:`get_sparse_joint_limit_rows_kernel` itself, Steel's builder for one word, so the limit rows are the
    ones the scene built before the internal unit; ``FAMILY_JOINT_FRICTION`` is
    :func:`get_sparse_joint_friction_rows_kernel`; ``FAMILY_VELOCITY_LIMIT`` is
    :func:`get_sparse_joint_velocity_limit_rows_kernel`, whose rows are late rows. ``fused`` is the drive family's
    alone; it raises :class:`ValueError` for another family. Raises :class:`ValueError` for the connect family, whose
    rows are not a factor group's (a loop closure's child may be a free body): its builder is
    :func:`get_scene_connect_rows_kernel`, through the contact response; and for an unknown family.
    """
    family = int(family)
    if fused and family != FAMILY_DRIVE:
        raise ValueError(f"fused is the drive family's build, got family {family}")
    if family == FAMILY_DRIVE:
        return get_sparse_drive_rows_kernel(mask_words, fused=fused)
    if family == FAMILY_LIMIT:
        return get_sparse_joint_limit_rows_kernel(mask_words)
    if family == FAMILY_JOINT_FRICTION:
        return get_sparse_joint_friction_rows_kernel(mask_words)
    if family == FAMILY_VELOCITY_LIMIT:
        return get_sparse_joint_velocity_limit_rows_kernel(mask_words)
    if family == FAMILY_MIMIC:
        return get_sparse_mimic_rows_kernel(mask_words)
    if family == FAMILY_CONNECT:
        raise ValueError(
            "the connect family's rows are built through the contact response, get_scene_connect_rows_kernel"
        )
    raise ValueError(f"family must be in [0, {FAMILY_COUNT}), got {family}")


CONNECT_NORM_MIN = 1.0e-8
"""Squared norm in joint coordinates below which a connect row is inert (zero ``J``, ``phi`` and target), Steel's
threshold (``kernels._finish_connect_row``)."""


@wp.func
def _joint_in_chain(joint: int, chain: int, joint_ancestor: wp.array[int]) -> bool:
    """Whether ``joint`` is ``chain`` or one of its ancestors; ``chain == -1`` is the empty chain."""
    found = bool(False)
    curr = int(chain)
    while curr != -1:
        if curr == joint:
            found = True
        curr = joint_ancestor[curr]
    return found


@wp.func
def connect_row_norms(
    joint_p: int,
    joint_c: int,
    rel_a: wp.vec3,
    rel_b: wp.vec3,
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
) -> wp.vec3:
    """The squared norms in joint coordinates of a loop closure's three rows, component ``e`` for world axis ``e``.

    Steel's row of axis ``e`` (``kernels.py`` ``populate_connect_J_for_size``) is ``J = e . (J_point(parent, p_A) -
    J_point(child, p_B))`` over the ancestor chains of the parent's joint ``joint_p`` (``-1`` for a prescribed
    parent, which contributes none) and of the child's ``joint_c``: DOF ``d`` of the parent's chain holds component
    ``e`` of ``lin + ang x rel_a``, of the child's ``-(lin + ang x rel_b)``, and of a shared ancestor their sum, with
    ``(lin, ang)`` its motion subspace ``joint_S_s[d]`` and ``rel_a``, ``rel_b`` the anchors less the articulation's
    origin. Steel normalizes each row by its norm (:data:`CONNECT_NORM_MIN`).
    """
    norms = wp.vec3(0.0)
    curr = int(joint_p)
    while curr != -1:
        shared = _joint_in_chain(curr, joint_c, joint_ancestor)
        for d in range(joint_qd_start[curr], joint_qd_start[curr + 1]):
            motion = joint_S_s[d]
            linear = wp.vec3(motion[0], motion[1], motion[2])
            angular = wp.vec3(motion[3], motion[4], motion[5])
            value = linear + wp.cross(angular, rel_a)
            if shared:
                value = value - (linear + wp.cross(angular, rel_b))
            norms += wp.cw_mul(value, value)
        curr = joint_ancestor[curr]
    curr = int(joint_c)
    while curr != -1:
        if not _joint_in_chain(curr, joint_p, joint_ancestor):
            for d in range(joint_qd_start[curr], joint_qd_start[curr + 1]):
                motion = joint_S_s[d]
                value = wp.vec3(motion[0], motion[1], motion[2]) + wp.cross(
                    wp.vec3(motion[3], motion[4], motion[5]), rel_b
                )
                norms += wp.cw_mul(value, value)
        curr = joint_ancestor[curr]
    return norms


CONNECT_ROWS_MAX = 6
"""The most rows of one loop closure (Steel's ``SolverSteel._connect_rows``): the three anchor rows of every closure,
then a REVOLUTE closure's two hinge rows or a FIXED closure's three weld rows (``kernels._connect_rotation_rows``)."""


@wp.func
def connect_rotation_row_norms(
    joint_p: int,
    joint_c: int,
    directions: wp.mat33,
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
) -> wp.vec3:
    """The squared norms in joint coordinates of a loop closure's rotation rows, component ``i`` for row ``i``.

    Steel's rotation row ``i`` (``kernels.populate_connect_J_for_size``) is ``J = e_i . (omega_parent -
    omega_child)``, ``e_i`` row ``i`` of ``directions`` (:func:`scene_connect_rotation_rows`), over the ancestor chains
    of the parent's joint ``joint_p`` (``-1`` for a prescribed parent, which contributes none) and of the child's
    ``joint_c``: DOF ``d`` of the parent's chain holds ``e_i . w``, of the child's ``-e_i . w``, with ``w`` the angular
    part of its motion subspace ``joint_S_s[d]``, and of a shared ancestor their sum, zero. Steel normalizes each row by
    its norm (:data:`CONNECT_NORM_MIN`).
    """
    norms = wp.vec3(0.0)
    curr = int(joint_p)
    while curr != -1:
        if not _joint_in_chain(curr, joint_c, joint_ancestor):
            for d in range(joint_qd_start[curr], joint_qd_start[curr + 1]):
                motion = joint_S_s[d]
                value = directions * wp.vec3(motion[3], motion[4], motion[5])
                norms += wp.cw_mul(value, value)
        curr = joint_ancestor[curr]
    curr = int(joint_c)
    while curr != -1:
        if not _joint_in_chain(curr, joint_p, joint_ancestor):
            for d in range(joint_qd_start[curr], joint_qd_start[curr + 1]):
                motion = joint_S_s[d]
                value = directions * wp.vec3(motion[3], motion[4], motion[5])
                norms += wp.cw_mul(value, value)
        curr = joint_ancestor[curr]
    return norms


@wp.func
def scene_connect_rotation_rows(
    rows: int,
    prescribed: bool,
    body_p: int,
    body_c: int,
    joint: int,
    joint_axis: wp.array[wp.vec3],
    joint_qd_start: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """A loop closure's rotation rows before their normalization, as Steel's ``kernels.populate_connect_J_for_size``.

    Returns ``(dirs, phis, targets)`` for the ``rows - 3`` rotation rows of a closure of ``rows`` rows (5 for a
    REVOLUTE closure's two hinge rows, 6 for a FIXED closure's three weld rows): row ``i`` of ``dirs`` is row ``i``'s
    direction ``e_i`` and ``phis[i]`` its position error, both from Steel's ``kernels._connect_rotation_rows``, and
    ``targets[i]`` its target, ``-e_i . omega_A`` under a ``prescribed`` parent body of angular velocity ``omega_A``,
    zero for the world and for a parent that responds. Its lines from the call to the target are Steel's
    (:data:`_SOURCE_CONNECT_ROTATION`), which a guard test compares with Steel's text.
    """
    dirs, phis = _connect_rotation_rows(
        rows, body_p, body_c, joint, joint_axis, joint_qd_start, joint_X_p, joint_X_c, body_q
    )
    omega_a = wp.vec3(0.0, 0.0, 0.0)
    if prescribed and body_p >= 0:
        omega_a = wp.spatial_bottom(body_qd[body_p])
    targets = wp.vec3(0.0)
    for i in range(rows - 3):
        e = dirs[i]
        target = float(0.0)
        if prescribed:
            target = -wp.dot(e, omega_a)
        targets[i] = target
    return dirs, phis, targets


_SOURCE_CONNECT_ROTATION = (
    ("dirs, phis = _connect_rotation_rows(", "omega_a = wp.spatial_bottom(body_qd[body_p])"),
    ("e = dirs[i]", "target = -wp.dot(e, omega_a)"),
)
"""The ranges of Steel's ``kernels.populate_connect_J_for_size`` that :func:`scene_connect_rotation_rows` restates, each
its first and last line (stripped, matched as prefixes, in order): the rotation rows' directions and errors with the
prescribed parent's angular velocity, and each row's target."""

_SOURCE_CONNECT_ROTATION_JACOBIAN = (
    "_accumulate_angular_jacobian_row(body_p,1.0,e,",
    "_accumulate_angular_jacobian_row(body_c,-1.0,e,",
)
"""The calls (whitespace removed) of Steel's ``kernels.populate_connect_J_for_size`` that build a rotation row's
Jacobian, ``e . omega_parent - e . omega_child``: the scene's connect rows give the response the couple ``e`` on both
sides, whose child side it subtracts."""


def _source_ranges(source: str, ranges: tuple[tuple[str, str], ...], name: str) -> list[str]:
    """The lines of ``source`` in ``ranges``, each range dedented on its own: the extractor of the guards that compare a
    restated law's lines with Steel's text (:func:`connect_rotation_mirror`, ``scene_torsion_device._mirrored_lines``).

    Each range is its first and last line, stripped and matched as prefixes, the ranges in order. Raises
    :class:`LookupError` naming ``name`` when a range is not in ``source``.
    """
    lines = source.splitlines()
    found = []
    position = 0
    for first, last in ranges:
        start = next((i for i in range(position, len(lines)) if lines[i].strip().startswith(first)), None)
        if start is None:
            raise LookupError(f"{name} has no line {first!r}")
        end = next((i for i in range(start, len(lines)) if lines[i].strip().startswith(last)), None)
        if end is None:
            raise LookupError(f"{name} has no line {last!r} after {first!r}")
        found.extend(textwrap.dedent("\n".join(lines[start : end + 1])).splitlines())
        position = end + 1
    return found


def connect_rotation_mirror(*, steel: str | None = None) -> tuple[list[str], list[str]]:
    """The restated lines of the connect rows' rotation law: :func:`scene_connect_rotation_rows`'s, then Steel's.

    Both are the lines of :data:`_SOURCE_CONNECT_ROTATION`'s ranges, in the scene's function and in Steel's
    ``kernels.populate_connect_J_for_size`` (or the source ``steel`` given in its place); equal lists mean the scene
    computes Steel's expressions, so a change of Steel's text fails the guard test that compares them. Raises
    :class:`LookupError` when a range is missing.
    """
    if steel is None:
        steel = inspect.getsource(populate_connect_J_for_size.func)
    scene = inspect.getsource(scene_connect_rotation_rows.func)
    return (
        _source_ranges(scene, _SOURCE_CONNECT_ROTATION, "scene_connect_rotation_rows"),
        _source_ranges(steel, _SOURCE_CONNECT_ROTATION, "populate_connect_J_for_size"),
    )


@cache
def get_scene_connect_rows_kernel(
    dof_max: int,
    mask_words: int = 1,
    *,
    max_closure_rows: int,
    lanes_per_contact: int = 8,
    warps_per_block: int = 4,
) -> wp.Kernel:
    """The scene's connect row builder: Steel's loop-closure rows through the contact response, for any factor group.

    One lane group per closure owner (``owner_art``, an articulation that owns a loop closure; its closures are
    ``owner_closures[owner_start[owner]:owner_start[owner + 1]]``, indices into the solver's connect table, in table
    order). Each closure whose joint is enabled (:attr:`~newton.Model.joint_enabled`, read every step) has its
    ``connect_rows[k]`` rows, Steel's (``kernels.py`` ``allocate_connect_slots`` and ``populate_connect_J_for_size``),
    type ``PGS_CONSTRAINT_TYPE_CONNECT``, no parent and no friction:

    - three anchor rows, one per world axis ``e``: ``J = e . (J_point(parent, p_A) - J_point(child, p_B))`` and
      ``phi = e . (p_A - p_B)``, the anchors the translations of the closure joint's ``joint_X_p`` and ``joint_X_c``
      placed by the bodies' poses;
    - then a REVOLUTE closure's two hinge rows or a FIXED closure's three weld rows: ``J = e_i . (omega_parent -
      omega_child)``, with the directions ``e_i`` and errors ``phi_i`` of Steel's ``kernels._connect_rotation_rows``
      (the normals of the live hinge axis, :func:`~newton._src.solvers.steel.kernels._connect_hinge_frame`, or the
      world axes and the weld's rotation vector) read every step (:func:`scene_connect_rotation_rows`).

    Each row is normalized by its norm in joint coordinates (:func:`connect_row_norms`,
    :func:`connect_rotation_row_norms`), ``phi`` and the target with it, and one whose squared norm is at most
    :data:`CONNECT_NORM_MIN` (or not a number) is inert (zero ``J``, ``phi`` and target), as Steel's. A prescribed
    parent (``connect_parent_prescribed``: the world, or a kinematic body of another articulation) contributes no DOFs;
    its anchor velocity enters the anchor rows' target, ``-e . v_A``, and its angular velocity the rotation rows',
    ``-e_i . omega_A``, both from its body twist.

    The anchor rows act at Steel's shared anchor: when the parent is not prescribed and both bodies' ancestor walks
    have a DOF that is not kinematic (``kernels._walk_has_free_dof`` on ``kinematic_dof_mask``), both Jacobian points
    (and the row norms' lever arms) are the anchors' midpoint ``p_A - (p_A - p_B) / 2``, so the impulse pair adds no
    couple across an open closure; otherwise each body keeps its own anchor. ``phi`` stays ``e . (p_A - p_B)``. A
    closure records the choice in ``connect_at_midpoint[k]`` for the force report, as
    ``kernels.populate_connect_J_for_size`` does, and with it the force report's other two inputs, as Steel's
    ``allocate_connect_slots`` and ``populate_connect_J_for_size`` do: its first row in ``connect_slot[k]`` (``-1`` for
    a released closure and one past the row capacity) and each row's normalization factor in
    ``connect_row_inv_norm[k, row]`` (zero for an inert row), which
    ``kernels.accumulate_connect_wrenches_from_impulses`` reads for ``BODY_PARENT_F``.

    ``max_closure_rows`` (required, no default) is the most rows a closure of the caller's connect table has
    (``max(connect_rows)``): 3 when every closure is a BALL joint, which leaves the rotation rows' code out of the
    kernel, 5 or :data:`CONNECT_ROWS_MAX` when a REVOLUTE or FIXED closure is present (one kernel for both). Any other
    count raises :class:`ValueError`, and a call that omits it raises :class:`TypeError`, so no caller builds a
    kernel without the rotation rows for a REVOLUTE or FIXED closure by default.

    The rows are projected by the contact response's native body, :func:`scene_contact_response_snippet` unchanged
    (the projection is the contacts' bit for bit), three anchor rows and then the rotation rows in a second call:
    directions ``e / |J|``, ``point_a = p_A`` and ``point_b = p_B`` for the anchor rows, and for the rotation rows no
    direction and the couple ``e_i / |J|`` on both sides, the response subtracting the child's; ``art_a == art_b`` the
    owner, the self-contact path that takes the OR of the two endpoints' support masks (``art_a = body_a = -1`` for a
    prescribed parent). A free-body child takes the response's free-body branch.

    ``ROW_SLOTS_COUNT`` adds each owner's rows to ``element_row_cursor`` and writes none; ``ROW_SLOTS_WRITE`` writes
    them from ``element_row_cursor[art]`` in closure order, each closure's rows in Steel's order, advancing it, and
    none of a closure whose rows would pass the row capacity (Steel drops such a closure whole; a dropped unit's cursor
    stands at the capacity). The arguments after ``articulation_dof_start`` are those of
    :func:`_get_scene_contact_response_kernel`'s group table and outputs, ``connect_at_midpoint``, ``connect_slot`` and
    ``connect_row_inv_norm`` after them; launch ``owners * lanes_per_contact`` threads in blocks of
    ``32 * warps_per_block``.
    """
    closure_rows_max = int(max_closure_rows)
    if closure_rows_max not in (3, CONNECT_ROWS_MAX - 1, CONNECT_ROWS_MAX):
        raise ValueError(
            f"max_closure_rows must be 3, {CONNECT_ROWS_MAX - 1} or {CONNECT_ROWS_MAX}, got {max_closure_rows}"
        )
    if closure_rows_max == CONNECT_ROWS_MAX - 1:
        # A REVOLUTE closure's rows are built by the kernel that builds a FIXED closure's.
        return get_scene_connect_rows_kernel(
            dof_max,
            mask_words,
            max_closure_rows=CONNECT_ROWS_MAX,
            lanes_per_contact=lanes_per_contact,
            warps_per_block=warps_per_block,
        )
    rotation = closure_rows_max > 3
    snippet = scene_contact_response_snippet(
        dof_max, mask_words, lanes_per_contact=lanes_per_contact, warps_per_block=warps_per_block
    )
    n, words, lanes, warps = int(dof_max), int(mask_words), int(lanes_per_contact), int(warps_per_block)

    @wp.func_native(snippet)
    def project_native(
        tid: int,
        body_a: int,
        body_b: int,
        art_a: int,
        art_b: int,
        world: int,
        slot: int,
        row_count: int,
        angular_a: wp.mat33,
        angular_b: wp.mat33,
        directions: wp.mat33,
        art_group_idx: wp.array[int],
        articulation_response_dof_count: wp.array[int],
        is_free_rigid: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        body_dof_mask: wp.array[wp.uint64],
        joint_S_s: wp.array[wp.spatial_vector],
        art_group: wp.array[int],
        art_slot: wp.array[int],
        group_dof_count: wp.array[int],
        group_nonzero_count: wp.array[int],
        group_linv_offset: wp.array[int],
        group_permutation_base: wp.array[int],
        group_row_offset_base: wp.array[int],
        permutation: wp.array[int],
        row_offsets: wp.array[int],
        Linv: wp.array[float],
        free_inverse_factor: wp.array3d[float],
        v_hat: wp.array[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
    ): ...

    def connect_rows_kernel(  # pragma: no cover - compiled by wp.kernel
        owner_art: wp.array[int],
        owner_start: wp.array[int],
        owner_closures: wp.array[int],
        connect_joint: wp.array[int],
        connect_body_p: wp.array[int],
        connect_body_c: wp.array[int],
        connect_parent_prescribed: wp.array[int],
        connect_rows: wp.array[int],
        kinematic_dof_mask: wp.array[int],
        joint_enabled: wp.array[bool],
        joint_axis: wp.array[wp.vec3],
        joint_X_p: wp.array[wp.transform],
        joint_X_c: wp.array[wp.transform],
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
        body_com: wp.array[wp.vec3],
        body_to_joint: wp.array[int],
        joint_ancestor: wp.array[int],
        joint_qd_start: wp.array[int],
        art_group_idx: wp.array[int],
        articulation_response_dof_count: wp.array[int],
        is_free_rigid: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        articulation_origin: wp.array[wp.vec3],
        body_dof_mask: wp.array[wp.uint64],
        joint_S_s: wp.array[wp.spatial_vector],
        art_group: wp.array[int],
        art_slot: wp.array[int],
        group_dof_count: wp.array[int],
        group_nonzero_count: wp.array[int],
        group_linv_offset: wp.array[int],
        group_permutation_base: wp.array[int],
        group_row_offset_base: wp.array[int],
        permutation: wp.array[int],
        row_offsets: wp.array[int],
        Linv: wp.array[float],
        free_inverse_factor: wp.array3d[float],
        v_hat: wp.array[float],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        phi: wp.array2d[float],
        target_velocity: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
        connect_at_midpoint: wp.array[int],
        connect_slot: wp.array[int],
        connect_row_inv_norm: wp.array2d[float],
        row_slot_mode: int,
        element_row_cursor: wp.array[int],
    ):
        tid = wp.tid()
        owner = tid // lanes
        if owner >= owner_art.shape[0]:
            return
        art = owner_art[owner]
        capacity = row_type.shape[1]
        # One lane group owns the articulation: every lane reads its cursor, lane 0 writes it back at the end.
        row = element_row_cursor[art]
        for index in range(owner_start[owner], owner_start[owner + 1]):
            k = owner_closures[index]
            joint = connect_joint[k]
            enabled = joint_enabled[joint]
            closure_rows = connect_rows[k]
            slot = row
            if enabled:
                row += closure_rows
            # A COUNT pass only advances the cursor. A WRITE pass records each closure's first row for the force
            # report, -1 for a released closure and for one whose rows would pass the capacity: as Steel's slot
            # allocation, it is dropped whole.
            if row_slot_mode == ROW_SLOTS_COUNT:
                continue
            written = enabled and slot + closure_rows <= capacity
            if tid % lanes == 0:
                if written:
                    connect_slot[k] = slot
                else:
                    connect_slot[k] = -1
            if not written:
                continue
            body_p = connect_body_p[k]
            body_c = connect_body_c[k]
            prescribed = connect_parent_prescribed[k]
            # A prescribed parent has no response: the parent side of the projection is empty.
            body_a = int(-1)
            art_a = int(-1)
            if prescribed == 0:
                body_a = body_p
                art_a = art
            angular_a = wp.mat33(0.0)
            angular_b = wp.mat33(0.0)
            directions = wp.mat33(0.0)
            row_phi = wp.vec3(0.0)
            row_target = wp.vec3(0.0)
            if tid % lanes == 0:
                anchor_p = wp.transform_get_translation(joint_X_p[joint])
                point_a = anchor_p
                velocity_a = wp.vec3(0.0)
                if body_p >= 0:
                    point_a = wp.transform_point(body_q[body_p], anchor_p)
                    if prescribed != 0:
                        # The anchor velocity of a prescribed parent comes from its body twist.
                        twist = body_qd[body_p]
                        com = wp.transform_point(body_q[body_p], body_com[body_p])
                        velocity_a = wp.spatial_top(twist) + wp.cross(wp.spatial_bottom(twist), point_a - com)
                point_b = wp.transform_point(body_q[body_c], wp.transform_get_translation(joint_X_c[joint]))
                origin = articulation_origin[art]
                delta = point_a - point_b
                # Steel's shared anchor: both bodies' rows act at the anchors' midpoint when both walks respond.
                at_midpoint = (
                    prescribed == 0
                    and _walk_has_free_dof(body_p, body_to_joint, joint_ancestor, joint_qd_start, kinematic_dof_mask)
                    and _walk_has_free_dof(body_c, body_to_joint, joint_ancestor, joint_qd_start, kinematic_dof_mask)
                )
                connect_at_midpoint[k] = int(at_midpoint)
                lever_a = point_a
                lever_b = point_b
                if at_midpoint:
                    lever_a = point_a - 0.5 * delta
                    lever_b = lever_a
                joint_p = int(-1)
                if prescribed == 0:
                    joint_p = body_to_joint[body_p]
                norms = connect_row_norms(
                    joint_p,
                    body_to_joint[body_c],
                    lever_a - origin,
                    lever_b - origin,
                    joint_ancestor,
                    joint_qd_start,
                    joint_S_s,
                )
                for axis in range(3):
                    # Steel's normalized row; a degenerate one stays inert (zero J, phi and target).
                    inv_norm = float(0.0)
                    if norms[axis] > CONNECT_NORM_MIN:
                        scale = 1.0 / wp.sqrt(norms[axis])
                        direction = wp.vec3(0.0)
                        direction[axis] = scale
                        # J = [direction, (point - origin) x direction] dot motion, as the contact rows'.
                        wrench_a = wp.vec3(0.0)
                        if art_a >= 0:
                            wrench_a = wp.cross(lever_a - origin, direction)
                        wrench_b = wp.cross(lever_b - origin, direction)
                        for column in range(3):
                            angular_a[axis, column] = wrench_a[column]
                            angular_b[axis, column] = wrench_b[column]
                            directions[axis, column] = direction[column]
                        row_phi[axis] = delta[axis] * scale
                        row_target[axis] = -velocity_a[axis] * scale
                        inv_norm = scale
                    # The factor maps the row's impulse back to a force (Steel's connect_row_inv_norm).
                    connect_row_inv_norm[k, axis] = inv_norm
            project_native(
                tid,
                body_a,
                body_c,
                art_a,
                art,
                0,
                slot,
                3,
                angular_a,
                angular_b,
                directions,
                art_group_idx,
                articulation_response_dof_count,
                is_free_rigid,
                articulation_world_dof_offset,
                articulation_dof_start,
                body_dof_mask,
                joint_S_s,
                art_group,
                art_slot,
                group_dof_count,
                group_nonzero_count,
                group_linv_offset,
                group_permutation_base,
                group_row_offset_base,
                permutation,
                row_offsets,
                Linv,
                free_inverse_factor,
                v_hat,
                row_dof,
                row_factor,
                row_free_response,
                row_incident,
                diagonal,
            )
            if tid % lanes == 0:
                for axis in range(3):
                    row_type[0, slot + axis] = PGS_CONSTRAINT_TYPE_CONNECT
                    row_parent[0, slot + axis] = -1
                    row_mu[0, slot + axis] = 0.0
                    phi[0, slot + axis] = row_phi[axis]
                    target_velocity[0, slot + axis] = row_target[axis]
            if wp.static(rotation):
                if closure_rows > 3:
                    # The rotation rows are pure couples: no direction, the couple e_i / |J| on both sides.
                    couple_a = wp.mat33(0.0)
                    couple_b = wp.mat33(0.0)
                    rotation_phi = wp.vec3(0.0)
                    rotation_target = wp.vec3(0.0)
                    if tid % lanes == 0:
                        dirs, phis, targets = scene_connect_rotation_rows(
                            closure_rows,
                            prescribed != 0,
                            body_p,
                            body_c,
                            joint,
                            joint_axis,
                            joint_qd_start,
                            joint_X_p,
                            joint_X_c,
                            body_q,
                            body_qd,
                        )
                        rotation_joint_p = int(-1)
                        if prescribed == 0:
                            rotation_joint_p = body_to_joint[body_p]
                        rotation_norms = connect_rotation_row_norms(
                            rotation_joint_p, body_to_joint[body_c], dirs, joint_ancestor, joint_qd_start, joint_S_s
                        )
                        for i in range(closure_rows - 3):
                            # Steel's normalized row; a degenerate one (or a non-finite hinge axis) stays inert.
                            inv_norm = float(0.0)
                            if rotation_norms[i] > CONNECT_NORM_MIN:
                                inv_norm = 1.0 / wp.sqrt(rotation_norms[i])
                                for column in range(3):
                                    couple_b[i, column] = dirs[i, column] * inv_norm
                                    if art_a >= 0:
                                        couple_a[i, column] = dirs[i, column] * inv_norm
                                rotation_phi[i] = phis[i] * inv_norm
                                rotation_target[i] = targets[i] * inv_norm
                            connect_row_inv_norm[k, 3 + i] = inv_norm
                    project_native(
                        tid,
                        body_a,
                        body_c,
                        art_a,
                        art,
                        0,
                        slot + 3,
                        closure_rows - 3,
                        couple_a,
                        couple_b,
                        wp.mat33(0.0),
                        art_group_idx,
                        articulation_response_dof_count,
                        is_free_rigid,
                        articulation_world_dof_offset,
                        articulation_dof_start,
                        body_dof_mask,
                        joint_S_s,
                        art_group,
                        art_slot,
                        group_dof_count,
                        group_nonzero_count,
                        group_linv_offset,
                        group_permutation_base,
                        group_row_offset_base,
                        permutation,
                        row_offsets,
                        Linv,
                        free_inverse_factor,
                        v_hat,
                        row_dof,
                        row_factor,
                        row_free_response,
                        row_incident,
                        diagonal,
                    )
                    if tid % lanes == 0:
                        for i in range(closure_rows - 3):
                            row_type[0, slot + 3 + i] = PGS_CONSTRAINT_TYPE_CONNECT
                            row_parent[0, slot + 3 + i] = -1
                            row_mu[0, slot + 3 + i] = 0.0
                            phi[0, slot + 3 + i] = rotation_phi[i]
                            target_velocity[0, slot + 3 + i] = rotation_target[i]
        if tid % lanes == 0:
            element_row_cursor[art] = row

    connect_rows_kernel.__name__ = (
        f"populate_scene_connect_rows_{n}_{lanes}_{warps}"
        + ("_rotation" if rotation else "")
        + (f"_w{words}" if words > 1 else "")
    )
    connect_rows_kernel.__qualname__ = connect_rows_kernel.__name__
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(connect_rows_kernel)


# --- Row primitives the scene modules share: row finishing and the velocity rows' row dot ----------------------


@wp.func
def _scene_row_free_prefix(row_dof: wp.array3d[int], free_dof_mask: wp.array2d[int], row: int) -> int:
    """Row ``row``'s leading entries in physical free-body coordinates: 0, 6 or 12.

    The response kernel puts a row's free endpoints first, six entries each, so the prefix is 6 when its first entry
    is a free body's coordinate and 12 when its seventh is one too.
    """
    prefix = int(0)
    first = row_dof[0, row, 0]
    if first >= 0:
        if free_dof_mask[0, first] != 0:
            prefix = 6
            if row_dof.shape[2] > 6:
                second = row_dof[0, row, 6]
                if second >= 0:
                    if free_dof_mask[0, second] != 0:
                        prefix = 12
    return prefix


@wp.func
def _finish_scene_row(
    row: int,
    bias: float,
    weight: float,
    contact_w: float,
    row_dof: wp.array3d[int],
    free_dof_mask: wp.array2d[int],
    rhs: wp.array2d[float],
    row_w: wp.array2d[float],
    impulses: wp.array2d[float],
    row_free_prefix: wp.array[int],
):
    """Write a row's right-hand side (its bias), weight (with contact regularization only), zero impulse and free
    prefix (:func:`_scene_row_free_prefix`): the tail of :func:`prepare_scene_rows`, which a spin row written after it
    shares (:func:`scene_torsion_finish`). The ``cfm`` on the diagonal stays with each caller: drive rows take none.
    """
    rhs[0, row] = bias
    if contact_w < 1.0:
        row_w[0, row] = weight
    impulses[0, row] = 0.0
    row_free_prefix[row] = _scene_row_free_prefix(row_dof, free_dof_mask, row)


_TREE_LANES = wp.types.vector(length=16, dtype=wp.float32)
"""The 16 partial sums of Steel's 32-lane tree after its first level (lanes ``l`` and ``l + 16``)."""


@wp.func
def _scene_row_dot(
    row_dof: wp.array3d[int], row_factor: wp.array3d[float], delta: wp.array2d[float], i: int, steel_sums: int
) -> float:
    """``Z delta`` of row ``i``: its factor values dotted with the velocity delta, summed as the sweep sums ``dot``.

    The one-thread row loop's arithmetic (:data:`SCENE_ROWS_CUDA`): with ``steel_sums`` Steel's 32-lane tree
    (entries ``l`` and ``l + 16`` in lane ``l``, each product rounded alone, then the levels 8, 4, 2 and 1), otherwise
    one multiply-add chain in entry order. A row's entries are a valid prefix padded with ``-1``; a free body's
    leading entries are physical coordinates, where the factor value is ``J`` itself and ``delta`` the velocity
    change.
    """
    support = row_dof.shape[2]
    if steel_sums != 0:
        lanes = _TREE_LANES()
        for lane in range(16):
            low = float(0.0)
            high = float(0.0)
            if lane < support:
                index = row_dof[0, i, lane]
                if index >= 0:
                    low += row_factor[0, i, lane] * delta[0, index]
            if lane + 16 < support:
                index = row_dof[0, i, lane + 16]
                if index >= 0:
                    high += row_factor[0, i, lane + 16] * delta[0, index]
            lanes[lane] = low + high
        for lane in range(8):
            lanes[lane] = lanes[lane] + lanes[lane + 8]
        for lane in range(4):
            lanes[lane] = lanes[lane] + lanes[lane + 4]
        for lane in range(2):
            lanes[lane] = lanes[lane] + lanes[lane + 2]
        return lanes[0] + lanes[1]
    dot = float(0.0)
    for k in range(support):
        index = row_dof[0, i, k]
        if index >= 0:
            dot += row_factor[0, i, k] * delta[0, index]
    return dot
