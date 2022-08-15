// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "enumerate_proposals_kernel.hpp"
#include <immintrin.h>

using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {

template <cpu_isa_t isa>
jit_uni_enumerate_proposals_kernel_f32<isa>::jit_uni_enumerate_proposals_kernel_f32(const jit_conf &jcp) :
    jit_uni_enumerate_proposals_kernel(jcp), JitKernelBase(jit_name()), reg_params{abi_param1} {
}

template <cpu_isa_t isa>
void jit_uni_enumerate_proposals_kernel_f32<isa>::generate() {
    preamble();
    Label h_loop_label;
    Label h_loop_end_label;
    Label w_loop_label;
    Label w_loop_end_label;
    Label feat_stride_const;
    Label simd_indices_offset_const;
    Label box_coordinate_scale_const;
    Label box_size_scale_const;
    Label coordinates_offset_const;
    Label half_value_const;

    mov(reg_bottom_h.cvt32(), dword[reg_params + offsetof(jit_call_args, bottom_H)]);
    mov(reg_bottom_w.cvt32(), dword[reg_params + offsetof(jit_call_args, bottom_W)]);
    xor_(reg_h, reg_h);
    Vmm reg_feat_stride { 1 };
//    uni_vbroadcastss(reg_feat_stride, ptr[rip + feat_stride_const]);
    uni_vmovups(reg_feat_stride, ptr[rip + feat_stride_const]);
    Vmm reg_simd_indices_offset { 2 };
    uni_vmovups(reg_simd_indices_offset, ptr[rip + simd_indices_offset_const]);
    L(h_loop_label);
    {
        cmp(reg_h, reg_bottom_h);
        jge(h_loop_end_label, T_NEAR);

        Vmm reg_h_fp32 { 3 };
        Vmm reg_w_fp32 { 4 };

        uni_vcvtsi2ss(Xmm(reg_h_fp32.getIdx()), Xmm(reg_h_fp32.getIdx()), reg_h.cvt32());
        uni_vbroadcastss(reg_h_fp32, Xmm(reg_h_fp32.getIdx()));
        uni_vmulps(reg_h_fp32, reg_h_fp32, reg_feat_stride);

        xor_(reg_w, reg_w);

        L(w_loop_label);
        {
            mov(rax, reg_bottom_w);
            sub(rax, simd_width);
            cmp(reg_w, rax);
            jge(w_loop_end_label, T_NEAR);

            uni_vcvtsi2ss(Xmm(reg_w_fp32.getIdx()), Xmm(reg_w_fp32.getIdx()), reg_w.cvt32());
            uni_vbroadcastss(reg_w_fp32, Xmm(reg_w_fp32.getIdx()));
            uni_vaddps(reg_w_fp32, reg_w_fp32, reg_simd_indices_offset);
            uni_vmulps(reg_w_fp32, reg_w_fp32, reg_feat_stride);

            // TODO: Optimize loading registers
            {
                Reg64 reg_anchor = r12;
                Reg64 reg_bottom_area = r13;
                Reg64 reg_p_box = r14;

                xor_(reg_anchor, reg_anchor);
                mov(Reg32{reg_anchor.getIdx()}, dword[reg_params + offsetof(jit_call_args, anchor)]);
                xor_(reg_bottom_area, reg_bottom_area);
                mov(Reg32{reg_bottom_area.getIdx()}, dword[reg_params + offsetof(jit_call_args, bottom_area)]);

//                const float *p_box = d_anchor4d + h * bottom_W + w;
                mov(rax, reg_h);
                mul(reg_bottom_w);
                add(rax, reg_w);
                mov(reg_p_box, sizeof(float));
                mul(reg_p_box);
                add(rax, ptr[reg_params + offsetof(jit_call_args, d_anchor4d)]);
                mov(reg_p_box, rax);

//                const float dx = p_box[(anchor * 4 + 0) * bottom_area] / box_coordinate_scale;
                Vmm dx { 5 };
                mov(rax, reg_anchor);
                mul(reg_bottom_area);
                // Multiply rax by 4
                shl(rax, 2);
                uni_vmovups(dx, ptr[reg_p_box + sizeof(float) * rax]);
                uni_vmulps(dx, dx, ptr[rip + box_coordinate_scale_const]);

//                const float dy = p_box[(anchor * 4 + 1) * bottom_area] / box_coordinate_scale;
                Vmm dy { 6 };
                lea(rax, ptr[1 + 4 * reg_anchor]);
                mul(reg_bottom_area);
                uni_vmovups(dy, ptr[reg_p_box + sizeof(float) * rax]);
                uni_vmulps(dy, dy, ptr[rip + box_coordinate_scale_const]);

                Vmm x { jcp_.swap_xy ? reg_h_fp32 : reg_w_fp32 };
                Vmm y { jcp_.swap_xy ? reg_w_fp32 : reg_h_fp32 };

//                    float x0 = x + p_anchors_wm[anchor];
                Vmm x0 { 7 };
                Vmm tmp { 11 };
                mov(rax, ptr[reg_params + offsetof(jit_call_args, p_anchors_wm)]);
                uni_vbroadcastss(tmp, ptr[rax + sizeof(float) * reg_anchor]);
                uni_vaddps(x0, x, tmp);

//                float y0 = y + p_anchors_hm[anchor];
                Vmm y0 { 8 };
                mov(rax, ptr[reg_params + offsetof(jit_call_args, p_anchors_hm)]);
                uni_vbroadcastss(tmp, ptr[rax + sizeof(float) * reg_anchor]);
                uni_vaddps(y0, y, tmp);

//                float x1 = x + p_anchors_wp[anchor];
                Vmm x1 { 9 };
                mov(rax, ptr[reg_params + offsetof(jit_call_args, p_anchors_wp)]);
                uni_vbroadcastss(tmp, ptr[rax + sizeof(float) * reg_anchor]);
                uni_vaddps(x1, x, tmp);

//                float y1 = y + p_anchors_hp[anchor];
                Vmm y1 { 10 };
                mov(rax, ptr[reg_params + offsetof(jit_call_args, p_anchors_hp)]);
                uni_vbroadcastss(tmp, ptr[rax + sizeof(float) * reg_anchor]);
                uni_vaddps(y1, y, tmp);

                if (jcp_.initial_clip) {
//                    // adjust new corner locations to be within the image region
                    Vmm reg_img_w { 11 };
                    uni_vbroadcastss(reg_img_w, ptr[reg_params + offsetof(jit_call_args, img_W)]);
                    Vmm reg_img_h { 12 };
                    uni_vbroadcastss(reg_img_h, ptr[reg_params + offsetof(jit_call_args, img_H)]);
                    Vmm reg_zero { 13 };
                    uni_vxorps(reg_zero, reg_zero, reg_zero);

//                    x0 = std::max<float>(0.0f, std::min<float>(x0, img_W));
                    uni_vminps(x0, x0, reg_img_w);
                    uni_vmaxps(x0, x0, reg_zero);
//                    y0 = std::max<float>(0.0f, std::min<float>(y0, img_H));
                    uni_vminps(y0, y0, reg_img_h);
                    uni_vmaxps(y0, y0, reg_zero);
//                    x1 = std::max<float>(0.0f, std::min<float>(x1, img_W));
                    uni_vminps(x1, x1, reg_img_w);
                    uni_vmaxps(x1, x1, reg_zero);
//                    y1 = std::max<float>(0.0f, std::min<float>(y1, img_H));
                    uni_vminps(y1, y1, reg_img_h);
                    uni_vmaxps(y1, y1, reg_zero);
                }

                // width & height of box
//                const float ww = x1 - x0 + coordinates_offset;
//                const float hh = y1 - y0 + coordinates_offset;
                Vmm ww { x1.getIdx() };
                Vmm hh { y1.getIdx() };
                uni_vsubps(ww, ww, x0);
                uni_vaddps(ww, ww, ptr[rip + coordinates_offset_const]);
                uni_vsubps(hh, hh, y0);
                uni_vaddps(hh, hh, ptr[rip + coordinates_offset_const]);
                // center location of box
//                const float ctr_x = x0 + 0.5f * ww;
//                const float ctr_y = y0 + 0.5f * hh;
                Vmm ctr_x { x0.getIdx() };
                Vmm ctr_y { y0.getIdx() };
                if (isValidIsa(avx2)) {
                    uni_vfmadd231ps(ctr_x, ww, ptr[rip + half_value_const]);
                    uni_vfmadd231ps(ctr_y, hh, ptr[rip + half_value_const]);
                } else {
                    Vmm tmp { 11 };
                    uni_vmovups(tmp, ptr[rip + half_value_const]);
                    uni_vmulps(tmp, tmp, ww);
                    uni_vaddps(ctr_x, ctr_x, tmp);
                    uni_vmovups(tmp, ptr[rip + half_value_const]);
                    uni_vmulps(tmp, tmp, hh);
                    uni_vaddps(ctr_y, ctr_y, tmp);
                }

                // new center location according to gradient (dx, dy)
//                const float pred_ctr_x = dx * ww + ctr_x;
//                const float pred_ctr_y = dy * hh + ctr_y;
                Vmm pred_ctr_x { ctr_x.getIdx() };
                Vmm pred_ctr_y { ctr_y.getIdx() };
                uni_vfmadd231ps(pred_ctr_x, dx, ww);
                uni_vfmadd231ps(pred_ctr_y, dy, hh);

//                const float d_log_w = p_box[(anchor * 4 + 2) * bottom_area] / box_size_scale;
                Vmm d_log_w { 11 };
                mov(rax, reg_anchor);
                // Multiply rax by 4
                shl(rax, 2);
                add(rax, 2);
                mul(reg_bottom_area);
                uni_vmovups(d_log_w, ptr[reg_p_box + sizeof(float) * rax]);
                uni_vmulps(d_log_w, d_log_w, ptr[rip + box_size_scale_const]);

//                const float d_log_h = p_box[(anchor * 4 + 3) * bottom_area] / box_size_scale;
                Vmm d_log_h { 12 };
                mov(rax, reg_anchor);
                // Multiply rax by 4
                shl(rax, 2);
                add(rax, 3);
                mul(reg_bottom_area);
                uni_vmovups(d_log_h, ptr[reg_p_box + sizeof(float) * rax]);
                uni_vmulps(d_log_h, d_log_h, ptr[rip + box_size_scale_const]);


                // new width & height according to gradient d(log w), d(log h)
//                const float pred_w = std::exp(d_log_w) * ww;
//                const float pred_h = std::exp(d_log_h) * hh;
                exp_injector_->compute_vector(d_log_w.getIdx());
                uni_vmulps(d_log_w, d_log_w, ww);
                exp_injector_->compute_vector(d_log_h.getIdx());
                uni_vmulps(d_log_h, d_log_h, hh);
                Vmm pred_w {d_log_w};
                Vmm pred_h {d_log_h};

                // update upper-left corner location
//                x1 = pred_ctr_x + 0.5f * pred_w;
//                y1 = pred_ctr_y + 0.5f * pred_h;
                uni_vmulps(pred_w, pred_w, ptr[rip + half_value_const]);
                uni_vmulps(pred_h, pred_h, ptr[rip + half_value_const]);
                uni_vaddps(x1, pred_ctr_x, pred_w);
                uni_vaddps(y1, pred_ctr_y, pred_h);
                // update lower-right corner location
//                x0 = pred_ctr_x - 0.5f * pred_w;
//                y0 = pred_ctr_y - 0.5f * pred_h;
                uni_vsubps(x0, pred_ctr_x, pred_w);
                uni_vsubps(y0, pred_ctr_y, pred_h);

                // adjust new corner locations to be within the image region,
                if (jcp_.clip_before_nms) {
                    Vmm reg_img_w { d_log_w.getIdx() };
                    uni_vbroadcastss(reg_img_w, ptr[reg_params + offsetof(jit_call_args, img_W)]);
                    uni_vsubps(reg_img_w, reg_img_w, ptr[rip + coordinates_offset_const]);
                    Vmm reg_img_h { d_log_h.getIdx() };
                    uni_vbroadcastss(reg_img_h, ptr[reg_params + offsetof(jit_call_args, img_H)]);
                    uni_vsubps(reg_img_h, reg_img_h, ptr[rip + coordinates_offset_const]);
                    Vmm reg_zero { 13 };
                    uni_vxorps(reg_zero, reg_zero, reg_zero);

//                    x0 = std::max<float>(0.0f, std::min<float>(x0, img_W - coordinates_offset));
                    uni_vminps(x0, x0, reg_img_w);
                    uni_vmaxps(x0, x0, reg_zero);
//                    y0 = std::max<float>(0.0f, std::min<float>(y0, img_H - coordinates_offset));
                    uni_vminps(y0, y0, reg_img_h);
                    uni_vmaxps(y0, y0, reg_zero);
//                    x1 = std::max<float>(0.0f, std::min<float>(x1, img_W - coordinates_offset));
                    uni_vminps(x1, x1, reg_img_w);
                    uni_vmaxps(x1, x1, reg_zero);
//                    y1 = std::max<float>(0.0f, std::min<float>(y1, img_H - coordinates_offset));
                    uni_vminps(y1, y1, reg_img_h);
                    uni_vmaxps(y1, y1, reg_zero);
                }

                // recompute new width & height
//                const float box_w = x1 - x0 + coordinates_offset;
//                const float box_h = y1 - y0 + coordinates_offset;
                Vmm box_w { dx.getIdx() };
                uni_vsubps(box_w, x1, x0);
                uni_vaddps(box_w, box_w, ptr[rip + coordinates_offset_const]);
                Vmm box_h { dy.getIdx() };
                uni_vsubps(box_h, y1, y0);
                uni_vaddps(box_h, box_h, ptr[rip + coordinates_offset_const]);

                Vmm score { d_log_w.getIdx() };

                if (isa == avx512_core) {
                    // uni_* wrappers don't allow passing opmask registers
                } else {
                    Vmm score_mask { 0 };
                    Vmm score_mask_w { score_mask.getIdx() };
                    Vmm score_mask_h { d_log_h.getIdx() };
                    {
                        Vmm min_box_W { 13 };
                        uni_vbroadcastss(min_box_W, ptr[reg_params + offsetof(jit_call_args, min_box_W)]);
                        uni_vcmpps(score_mask_w, min_box_W, box_w, _cmp_nle_us);
                        Vmm min_box_H { min_box_W.getIdx() };
                        uni_vbroadcastss(min_box_H, ptr[reg_params + offsetof(jit_call_args, min_box_H)]);
                        uni_vcmpps(score_mask_h, min_box_H, box_h, _cmp_nle_us);
                        uni_vorps(score_mask, score_mask_w, score_mask_h);
                    }
                    mov(eax, dword[reg_params + offsetof(jit_call_args, bottom_W)]);
                    mul(reg_h);
                    add(rax, reg_w);

                    mov(rbx, rax);
                    mov(rax, reg_anchor);
                    mul(reg_bottom_area);
                    add(rax, rbx);

                    mov(rbx, ptr[reg_params + offsetof(jit_call_args, bottom4d)]);
                    Vmm zero { 13 };
                    uni_vxorps(zero, zero, zero);
                    uni_vmovups(score, ptr[rbx + rax * sizeof(float)]);
                    uni_vblendvps(score, score, zero, score_mask);
                }

//                float *p_proposal = proposals + (h * bottom_W + w) * num_anchors * 5;
//                p_proposal[5 * anchor + 0] = x0;
//                p_proposal[5 * anchor + 1] = y0;
//                p_proposal[5 * anchor + 2] = x1;
//                p_proposal[5 * anchor + 3] = y1;
//                p_proposal[5 * anchor + 4] = (min_box_W <= box_w) * (min_box_H <= box_h) * score;

                Reg64 reg_h_offset { reg_bottom_area.getIdx() };
                mov(eax, dword[reg_params + offsetof(jit_call_args, bottom_W)]);
                mul(reg_h);
                mov(reg_h_offset, rax);
                Reg64 reg_anchor_offset { reg_p_box.getIdx() };
                mov(reg_anchor_offset, jcp_.num_anchors * 5 * sizeof(float));

                for (unsigned i = 0; i < simd_width; i++) {
                    mov(rax, reg_w);
                    add(rax, i);
                    add(rax, reg_h_offset);
                    mul(reg_anchor_offset);
                    add(rax, ptr[reg_params + offsetof(jit_call_args, proposals)]);
                    mov(rbx, rax);
                    mov(rax, 5);
                    mul(reg_anchor);
                    uni_vpextrd(ptr[rbx + rax * sizeof(float)], x0, i);
                    inc(rax);
                    uni_vpextrd(ptr[rbx + rax * sizeof(float)], y0, i);
                    inc(rax);
                    uni_vpextrd(ptr[rbx + rax * sizeof(float)], x1, i);
                    inc(rax);
                    uni_vpextrd(ptr[rbx + rax * sizeof(float)], y1, i);
                    inc(rax);
                    uni_vpextrd(ptr[rbx + rax * sizeof(float)], score, i);
                }
            }

            add(reg_w, simd_width);
            jmp(w_loop_label, T_NEAR);
        }
        L(w_loop_end_label);

        inc(reg_h);
        jmp(h_loop_label, T_NEAR);
    }
    L(h_loop_end_label);
    postamble();

    // Constants
    std::uint32_t mem_placeholder;

    L_aligned(simd_indices_offset_const);
    {
        for (int i = 0; i < simd_width; i++) {
            float index_offset { i };
            memcpy(&mem_placeholder, &index_offset, sizeof mem_placeholder);
            dd(mem_placeholder);
        }
    }
    L_aligned(box_coordinate_scale_const);
    {
        // Do division at compilation phase since multiplication performance is better
        float box_coordinate_scale = 1.f / jcp_.box_coordinate_scale;
        memcpy(&mem_placeholder, &box_coordinate_scale, sizeof(float));
        for (int i = 0; i < simd_width; i++)
            dd(mem_placeholder);
    }
    L_aligned(box_size_scale_const);
    {
        float box_size_scale = 1.f / jcp_.box_size_scale_;
        memcpy(&mem_placeholder, &box_size_scale, sizeof(float));
        for (int i = 0; i < simd_width; i++)
            dd(mem_placeholder);
    }
    L_aligned(coordinates_offset_const);
    {
        memcpy(&mem_placeholder, &jcp_.coordinates_offset, sizeof(float));
        for (int i = 0; i < simd_width; i++)
            dd(mem_placeholder);
    }
    L_aligned(half_value_const);
    {
        float half { 0.5f };
        memcpy(&mem_placeholder, &half, sizeof(float));
        for (int i = 0; i < simd_width; i++)
            dd(mem_placeholder);
    }
    L_aligned(feat_stride_const);
    {
        memcpy(&mem_placeholder, &jcp_.feat_stride, sizeof(float));
        for (int i = 0; i < simd_width; i++)
            dd(mem_placeholder);
    }
    exp_injector_->prepare_table();
}

template class jit_uni_enumerate_proposals_kernel_f32<sse41>;
template class jit_uni_enumerate_proposals_kernel_f32<avx2>;
template class jit_uni_enumerate_proposals_kernel_f32<avx512_core>;

} // namespace node
} // namespace intel_cpu
} // namespace ov
