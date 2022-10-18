// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/xpu/__xpu__qkv_attention_compute.h"
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

// void XPUQkvAttentionCompute::PrepareForRun() {
//   auto& param = this->template Param<param_t>();
//   auto* input = param.input; 
//   auto input_dims = input->dims();
//   int batch_size = static_cast<int>(input->dims()[0]);
//   int seq_len = input->dims()[1];
//   int head_num = param.head_num;
//   int qk_sum = 0;
//   for (int i = 0; i < batch_size; i++) {
//     qk_sum += head_num * seq_len * seq_len;
//   }
//   qk_guard_ = TargetWrapperXPU::MallocScratchPad(qk_sum * sizeof(float));
//   mask_guard_ = TargetWrapperXPU::MallocScratchPad(seq_len * seq_len * sizeof(float));
// }

void XPUQkvAttentionCompute::Run() {
    auto& param = this->Param<param_t>();
    auto& ctx = this->ctx_->As<XPUContext>();
    auto* input = param.input; 

    auto input_dims = input->dims();
    int batch_size = static_cast<int>(input->dims()[0]);
    int seq_len = input->dims()[1];  
    int qkv_len = input->dims()[2];  
    int dim = 3; // q k v
    int multi_head_size = qkv_len / dim;
    int head_num = param.head_num;
    int size_per_head = multi_head_size / head_num;
    // int r = 0; //ret value use

    const float* input_data = input->data<float>();
    const float* q = input_data;
    const float* k = q + multi_head_size;
    const float* v = q + 2 * multi_head_size;

    int qk_sum = 0;
    for (int i = 0; i < batch_size; i++) {
        qk_sum += head_num * seq_len * seq_len;
    }
    XPUScratchPadGuard internal_result_xpu_guard =
        TargetWrapperXPU::MallocScratchPad(qk_sum * sizeof(float));
    float* qk_out = reinterpret_cast<float*>(internal_result_xpu_guard->addr_);

    // XPUScratchPadGuard qk_guard = TargetWrapperXPU::MallocScratchPad(qk_sum * sizeof(float));
    // XPUScratchPadGuard mask_guard = TargetWrapperXPU::MallocScratchPad(seq_len * seq_len * sizeof(float));
    
    /* // some issue code by use for seting xx_max_guard 
    int maxptr_size = ctx.GetRawContext()->max_ptr_size();
    q_max_guard_ = TargetWrapperXPU::MallocScratchPad(maxptr_size * sizeof(float));
    k_max_guard_ = TargetWrapperXPU::MallocScratchPad(maxptr_size * sizeof(float));
    v_max_guard_ = TargetWrapperXPU::MallocScratchPad(maxptr_size * sizeof(float));
    qk_max_guard_ = TargetWrapperXPU::MallocScratchPad(maxptr_size * sizeof(float));
    qkv_max_guard_ = TargetWrapperXPU::MallocScratchPad(maxptr_size * sizeof(float));
    float* q_max_data = reinterpret_cast<float*>(q_max_guard_->addr_);
    float* k_max_data = reinterpret_cast<float*>(k_max_guard_->addr_);
    float* v_max_data = reinterpret_cast<float*>(v_max_guard_->addr_);
    float* qk_max_data = reinterpret_cast<float*>(qk_max_guard_->addr_);
    float* qkv_max_data = reinterpret_cast<float*>(qkv_max_guard_->addr_);
    
    r = xdnn::findmax<float>(ctx.GetRawContext(), q, q_max_data, multi_head_size * dim);
    CHECK_EQ(r, 0);
    r = xdnn::findmax<float>(ctx.GetRawContext(), k, k_max_data, multi_head_size * dim);
    CHECK_EQ(r, 0);
    r = xdnn::findmax<float>(ctx.GetRawContext(), v, v_max_data, multi_head_size * dim);
    CHECK_EQ(r, 0);
    */
    std::vector<int> mask = {1, 1, seq_len, seq_len};
    xdnn::QKVAttnParam qkv_attn_param(batch_size, 
                                      seq_len,
                                      head_num,
                                      size_per_head,
                                      mask,
                                      xdnn::Activation_t::RELU,
                                      -1,
                                      true,
                                      -1);

    // std::vector<int> lod_vec = {0};
    // for(int i = 1; i <= batch_size; ++i) {
    //     lod_vec.push_back(seq_len * i);
    // }
    // xdnn::VectorParam<int> lods = {lod_vec.data(), static_cast<int>(lod_vec.size()), nullptr};
    // xdnn::QKVAttnParam qkv_attn_param = {lods, head_num, size_per_head, xdnn::Activation_t::RELU, -1, false};

    /* some issue code use for dynamic seqlen debug
    xdnn::VectorParam<int> query_lod;
    std::vector<int> lod_cpu;
    lod_cpu.push_back(0);
    for (size_t i = 0; i < batch_size; i++){
        lod_cpu.push_back(lod_cpu.end() + );
    }

    if (param.SeqLod && param.SeqLod->data<int>()) {
        query_lod = {param.SeqLod->data<int>(),
                    static_cast<int>(param.SeqLod->numel()),
                    nullptr};
        // int max_pad_seqlen = slice_idx == -1 ? param.PadSeqLen->data<int>()[0] : -1;
        xdnn::QKVAttnParam qkv_attn_param(query_lod, 
                                        head_num,
                                        size_per_head,
                                        xdnn::Activation_t::RELU,
                                        -1,
                                        true);
    XPUScratchPadGuard xpu_mask =
        TargetWrapperXPU::MallocScratchPad(seq_len * seq_len * sizeof(float));
    */
    void* ptr{nullptr};
    XPU_CALL(xpu_malloc(&ptr, seq_len * seq_len * sizeof(float)));
    // XPUScratchPadGuard ptr =
    //     TargetWrapperXPU::MallocScratchPad(seq_len * seq_len * sizeof(float));
    // float* mask_ptr = reinterpret_cast<float*>(ptr->addr_);

    int r = xdnn::constant<float>(ctx.GetRawContext(), 
        reinterpret_cast<float*>(ptr), 
        seq_len * seq_len,
        0.0f);
    CHECK_EQ(r, 0);

    if (param.precision == "int31") {
        r = xdnn::qk_attention<float, float, float, int16_t>(
            ctx.GetRawContext(), 
            q,
            k,
            qk_out,
            nullptr,
            nullptr,
            nullptr,
            qkv_attn_param,
            reinterpret_cast<float*>(ptr)
        );
        XPU_CALL(xpu_free(ptr));
        ptr = nullptr;
        CHECK_EQ(r, 0);
        r = xdnn::qk_v_attention<float, float, float, int16_t>(
                ctx.GetRawContext(),                                          // ctx
                qk_out,                                                       // qk
                v,                                                            // v
                param.output->mutable_data<float>(TARGET(kXPU)),              // qkv
                nullptr,                                                      // max_qk
                nullptr,                                                      // max_v
                nullptr,                                                      // max_qkv
                qkv_attn_param                                                // p
                );
        CHECK_EQ(r, 0);
    } else {
        XPU_CALL(xpu_free(ptr));
        ptr = nullptr;
        CHECK(false);
    }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__qkv_attention,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUQkvAttentionCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
