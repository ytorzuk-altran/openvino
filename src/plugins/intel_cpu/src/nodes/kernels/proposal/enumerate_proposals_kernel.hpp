// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "../jit_kernel_base.hpp"
#include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>
#include <cassert>
#include <memory>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {
namespace details {
namespace x64 = dnnl::impl::cpu::x64;
} // namespace details

class jit_uni_enumerate_proposals_kernel {
public:
    struct jit_conf {
        size_t num_anchors;
        float feat_stride;
        float box_coordinate_scale;
        float box_size_scale_;
        float coordinates_offset;
        bool initial_clip;
        bool swap_xy;
        bool clip_before_nms;
    };

    struct jit_call_args {
        const float *bottom4d;
        const float *d_anchor4d;
        const float *p_anchors_wm;
        const float *p_anchors_hm;
        const float *p_anchors_wp;
        const float *p_anchors_hp;
        float *proposals;
        int anchor;
        int bottom_H;
        int bottom_W;
        int bottom_area;
        float img_H;
        float img_W;
        float min_box_H;
        float min_box_W;
    };

    void operator()(const jit_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_enumerate_proposals_kernel(jit_conf jcp) :
        jcp_(jcp) {
    }
    virtual ~jit_uni_enumerate_proposals_kernel() = default;

    virtual void create_ker() = 0;

    virtual unsigned simd_size() const = 0;

protected:
    void (*ker_)(const jit_call_args*) = nullptr;
    jit_conf jcp_;
};

template<typename Vmm>
const void* expfVecFuncAddr();

template <details::x64::cpu_isa_t isa>
struct jit_uni_enumerate_proposals_kernel_f32 : public jit_uni_enumerate_proposals_kernel, public JitKernelBase {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_enumerate_proposals_kernel_f32)

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    unsigned simd_size() const override {
        return simd_width;
    }

    explicit jit_uni_enumerate_proposals_kernel_f32(const jit_conf& jcp);

    using Vmm = typename dnnl::impl::utils::conditional3<isa == details::x64::sse41, Xbyak::Xmm, isa == details::x64::avx2,
                Xbyak::Ymm, Xbyak::Zmm>::type;

    void generate() override;

private:
    static constexpr unsigned simd_width = details::x64::cpu_isa_traits<isa>::vlen / sizeof(float);
    std::shared_ptr<details::x64::jit_uni_eltwise_injector_f32<isa>> exp_injector_ =
        std::make_shared<details::x64::jit_uni_eltwise_injector_f32<isa>>(this, dnnl::impl::alg_kind::eltwise_exp, 0.f, 0.f, 1.f);
    Xbyak::Reg64 reg_params;
};

} // namespace node
} // namespace intel_cpu
} // namespace ov
