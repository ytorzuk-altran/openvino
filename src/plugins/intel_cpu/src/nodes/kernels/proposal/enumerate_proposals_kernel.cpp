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
    registersPool = RegistersPool::create(isa, {rax, rdx, rbx, rsp, reg_params, k0});
    RegistersPool::Reg<Vmm> score_mask { registersPool, 0 };
    RegistersPool::Reg<Reg64> reg_h{ registersPool };
    RegistersPool::Reg<Reg64> reg_w{ registersPool };
    RegistersPool::Reg<Reg64> reg_bottom_h{ registersPool };
    RegistersPool::Reg<Reg64> reg_bottom_w{ registersPool };
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
    Label scatter_shift_const;

    mov(static_cast<Reg64>(reg_bottom_h).cvt32(), dword[reg_params + offsetof(jit_call_args, bottom_H)]);
    mov(static_cast<Reg64>(reg_bottom_w).cvt32(), dword[reg_params + offsetof(jit_call_args, bottom_W)]);
    xor_(reg_h, reg_h);
    RegistersPool::Reg<Vmm> reg_feat_stride { registersPool };
//    uni_vbroadcastss(reg_feat_stride, ptr[rip + feat_stride_const]);
    uni_vmovups(reg_feat_stride, ptr[rip + feat_stride_const]);
    RegistersPool::Reg<Vmm> reg_simd_indices_offset { registersPool };
    uni_vmovups(reg_simd_indices_offset, ptr[rip + simd_indices_offset_const]);
    L(h_loop_label);
    {
        cmp(reg_h, reg_bottom_h);
        jge(h_loop_end_label, T_NEAR);

        RegistersPool::Reg<Vmm> reg_h_fp32 { registersPool };
        RegistersPool::Reg<Vmm> reg_w_fp32 { registersPool };

        uni_vcvtsi2ss(Xmm(reg_h_fp32.getIdx()), Xmm(reg_h_fp32.getIdx()), static_cast<Reg64>(reg_h).cvt32());
        uni_vbroadcastss(reg_h_fp32, Xmm(reg_h_fp32.getIdx()));
        uni_vmulps(reg_h_fp32, reg_h_fp32, reg_feat_stride);

        xor_(reg_w, reg_w);

        L(w_loop_label);
        {
            mov(rax, reg_bottom_w);
            sub(rax, simd_width);
            cmp(reg_w, rax);
            jge(w_loop_end_label, T_NEAR);

            uni_vcvtsi2ss(Xmm(reg_w_fp32.getIdx()), Xmm(reg_w_fp32.getIdx()), static_cast<Reg64>(reg_w).cvt32());
            uni_vbroadcastss(reg_w_fp32, Xmm(reg_w_fp32.getIdx()));
            uni_vaddps(reg_w_fp32, reg_w_fp32, reg_simd_indices_offset);
            uni_vmulps(reg_w_fp32, reg_w_fp32, reg_feat_stride);

            {
                RegistersPool::Reg<Reg64> reg_anchor{registersPool};
                RegistersPool::Reg<Reg64> reg_bottom_area{registersPool};
                RegistersPool::Reg<Reg64> reg_p_box{registersPool};

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

                Vmm x { jcp_.swap_xy ? reg_h_fp32 : reg_w_fp32 };
                Vmm y { jcp_.swap_xy ? reg_w_fp32 : reg_h_fp32 };

                RegistersPool::Reg<Vmm> x0 { registersPool };
                RegistersPool::Reg<Vmm> y0 { registersPool };
                RegistersPool::Reg<Vmm> x1 { registersPool };
                RegistersPool::Reg<Vmm> y1 { registersPool };
                {
    //                const float dx = p_box[(anchor * 4 + 0) * bottom_area] / box_coordinate_scale;
                    RegistersPool::Reg<Vmm> dx { registersPool };
                    mov(rax, reg_anchor);
                    mul(reg_bottom_area);
                    // Multiply rax by 4
                    shl(rax, 2);
                    uni_vmovups(dx, ptr[reg_p_box + sizeof(float) * rax]);
                    uni_vmulps(dx, dx, ptr[rip + box_coordinate_scale_const]);

    //                const float dy = p_box[(anchor * 4 + 1) * bottom_area] / box_coordinate_scale;
                    RegistersPool::Reg<Vmm> dy { registersPool };
                    lea(rax, ptr[1 + 4 * reg_anchor]);
                    mul(reg_bottom_area);
                    uni_vmovups(dy, ptr[reg_p_box + sizeof(float) * rax]);
                    uni_vmulps(dy, dy, ptr[rip + box_coordinate_scale_const]);

                    {
        //                    float x0 = x + p_anchors_wm[anchor];
                        RegistersPool::Reg<Vmm> tmp { registersPool };
                        mov(rax, ptr[reg_params + offsetof(jit_call_args, p_anchors_wm)]);
                        uni_vbroadcastss(tmp, ptr[rax + sizeof(float) * reg_anchor]);
                        uni_vaddps(x0, x, tmp);

        //                float y0 = y + p_anchors_hm[anchor];
                        mov(rax, ptr[reg_params + offsetof(jit_call_args, p_anchors_hm)]);
                        uni_vbroadcastss(tmp, ptr[rax + sizeof(float) * reg_anchor]);
                        uni_vaddps(y0, y, tmp);

        //                float x1 = x + p_anchors_wp[anchor];
                        mov(rax, ptr[reg_params + offsetof(jit_call_args, p_anchors_wp)]);
                        uni_vbroadcastss(tmp, ptr[rax + sizeof(float) * reg_anchor]);
                        uni_vaddps(x1, x, tmp);

        //                float y1 = y + p_anchors_hp[anchor];
                        mov(rax, ptr[reg_params + offsetof(jit_call_args, p_anchors_hp)]);
                        uni_vbroadcastss(tmp, ptr[rax + sizeof(float) * reg_anchor]);
                        uni_vaddps(y1, y, tmp);
                    }

                    if (jcp_.initial_clip) {
    //                    // adjust new corner locations to be within the image region
                        RegistersPool::Reg<Vmm> reg_img_w { registersPool };
                        uni_vbroadcastss(reg_img_w, ptr[reg_params + offsetof(jit_call_args, img_W)]);
                        RegistersPool::Reg<Vmm> reg_img_h { registersPool };
                        uni_vbroadcastss(reg_img_h, ptr[reg_params + offsetof(jit_call_args, img_H)]);
                        RegistersPool::Reg<Vmm> reg_zero { registersPool };
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

                    {
                        RegistersPool::Reg<Vmm> pred_w;
                        RegistersPool::Reg<Vmm> pred_h;
                        RegistersPool::Reg<Vmm> pred_ctr_x;
                        RegistersPool::Reg<Vmm> pred_ctr_y;
                        {
                            // width & height of box
            //                const float ww = x1 - x0 + coordinates_offset;
            //                const float hh = y1 - y0 + coordinates_offset;
                            RegistersPool::Reg<Vmm> ww { std::move(x1) };
                            RegistersPool::Reg<Vmm> hh { std::move(y1) };
                            uni_vsubps(ww, ww, x0);
                            uni_vaddps(ww, ww, ptr[rip + coordinates_offset_const]);
                            uni_vsubps(hh, hh, y0);
                            uni_vaddps(hh, hh, ptr[rip + coordinates_offset_const]);
                            // center location of box
            //                const float ctr_x = x0 + 0.5f * ww;
            //                const float ctr_y = y0 + 0.5f * hh;
                            RegistersPool::Reg<Vmm> ctr_x { std::move(x0) };
                            RegistersPool::Reg<Vmm> ctr_y { std::move(y0) };
                            if (isValidIsa(avx2)) {
                                uni_vfmadd231ps(ctr_x, ww, ptr[rip + half_value_const]);
                                uni_vfmadd231ps(ctr_y, hh, ptr[rip + half_value_const]);
                            } else {
                                RegistersPool::Reg<Vmm> tmp { registersPool };
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
                            pred_ctr_x = std::move(ctr_x);
                            pred_ctr_y = std::move(ctr_y);
                            uni_vfmadd231ps(pred_ctr_x, dx, ww);
                            uni_vfmadd231ps(pred_ctr_y, dy, hh);

            //                const float d_log_w = p_box[(anchor * 4 + 2) * bottom_area] / box_size_scale;
                            RegistersPool::Reg<Vmm> d_log_w { registersPool };
                            mov(rax, reg_anchor);
                            // Multiply rax by 4
                            shl(rax, 2);
                            add(rax, 2);
                            mul(reg_bottom_area);
                            uni_vmovups(d_log_w, ptr[reg_p_box + sizeof(float) * rax]);
                            uni_vmulps(d_log_w, d_log_w, ptr[rip + box_size_scale_const]);

            //                const float d_log_h = p_box[(anchor * 4 + 3) * bottom_area] / box_size_scale;
                            RegistersPool::Reg<Vmm> d_log_h { registersPool };
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

                            pred_w = std::move(d_log_w);
                            pred_h = std::move(d_log_h);
                            x1 = std::move(ww);
                            y1 = std::move(hh);
                        }

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
                        uni_vsubps(pred_ctr_x, pred_ctr_x, pred_w);
                        uni_vsubps(pred_ctr_y, pred_ctr_y, pred_h);

                        x0 = std::move(pred_ctr_x);
                        y0 = std::move(pred_ctr_y);
                    }
                }

                // adjust new corner locations to be within the image region,
                if (jcp_.clip_before_nms) {
                    RegistersPool::Reg<Vmm> reg_img_w { registersPool };
                    uni_vbroadcastss(reg_img_w, ptr[reg_params + offsetof(jit_call_args, img_W)]);
                    uni_vsubps(reg_img_w, reg_img_w, ptr[rip + coordinates_offset_const]);
                    RegistersPool::Reg<Vmm> reg_img_h { registersPool };
                    uni_vbroadcastss(reg_img_h, ptr[reg_params + offsetof(jit_call_args, img_H)]);
                    uni_vsubps(reg_img_h, reg_img_h, ptr[rip + coordinates_offset_const]);
                    RegistersPool::Reg<Vmm> reg_zero { registersPool };
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
                RegistersPool::Reg<Vmm> box_w { registersPool };
                uni_vsubps(box_w, x1, x0);
                uni_vaddps(box_w, box_w, ptr[rip + coordinates_offset_const]);
                RegistersPool::Reg<Vmm> box_h { registersPool };
                uni_vsubps(box_h, y1, y0);
                uni_vaddps(box_h, box_h, ptr[rip + coordinates_offset_const]);

                RegistersPool::Reg<Vmm> score { registersPool };

                if (isa == avx512_core) {
                    // uni_* wrappers don't allow passing opmask registers
                } else {
                    {
                        Vmm score_mask_w { score_mask };
                        {
                            RegistersPool::Reg<Vmm> min_box_W { registersPool };
                            uni_vbroadcastss(min_box_W, ptr[reg_params + offsetof(jit_call_args, min_box_W)]);
                            uni_vcmpps(score_mask_w, min_box_W, box_w, _cmp_nle_us);
                        }
                        RegistersPool::Reg<Vmm> score_mask_h { registersPool };
                        {
                            RegistersPool::Reg<Vmm> min_box_H { registersPool };
                            uni_vbroadcastss(min_box_H, ptr[reg_params + offsetof(jit_call_args, min_box_H)]);
                            uni_vcmpps(score_mask_h, min_box_H, box_h, _cmp_nle_us);
                        }
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
                    RegistersPool::Reg<Vmm> zero { registersPool };
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

                RegistersPool::Reg<Reg64> reg_anchor_offset { std::move(reg_p_box) };
                mov(reg_anchor_offset, jcp_.num_anchors * 5 * sizeof(float));

                mov(rax, 5 * sizeof(float));
                mul(reg_anchor);
                mov(rbx, rax);

                mov(eax, reg_bottom_w);
                mul(reg_h);
                add(rax, reg_w);
                mul(reg_anchor_offset);
                add(rax, ptr[reg_params + offsetof(jit_call_args, proposals)]);
                add(rax, rbx);

                Vmm shift { score_mask };
                uni_vmovups(shift, ptr[rip + scatter_shift_const]);
                scatterdd(rax, static_cast<Vmm&>(x0), shift, k1);
                add(rax, sizeof(float));
                scatterdd(rax, static_cast<Vmm>(y0), shift, k1);
                add(rax, sizeof(float));
                scatterdd(rax, static_cast<Vmm>(x1), shift, k1);
                add(rax, sizeof(float));
                scatterdd(rax, static_cast<Vmm>(y1), shift, k1);
                add(rax, sizeof(float));
                scatterdd(rax, static_cast<Vmm>(score), shift, k1);
            }

            add(reg_w, simd_width);
            jmp(w_loop_label, T_NEAR);
        }
        L(w_loop_end_label);

        inc(reg_h);
        jmp(h_loop_label, T_NEAR);
    }
    L(h_loop_end_label);
    registersPool.reset();
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
    L_aligned(scatter_shift_const);
    {
        for (std::uint32_t i = 0; i < simd_width; i++) {
            std::uint32_t shift = i * jcp_.num_anchors * sizeof(float) * 5;
            dd(shift);
        }
    }
    exp_injector_->prepare_table();
}

//template class jit_uni_enumerate_proposals_kernel_f32<sse41>;
template class jit_uni_enumerate_proposals_kernel_f32<avx2>;
template class jit_uni_enumerate_proposals_kernel_f32<avx512_core>;

} // namespace node
} // namespace intel_cpu
} // namespace ov
