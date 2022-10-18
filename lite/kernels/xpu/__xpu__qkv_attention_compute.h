/*
 * @Author: AlbertVan ncepuzf11@163.com
 * @Date: 2022-09-20 22:39:57
 * @LastEditors: AlbertVan ncepuzf11@163.com
 * @LastEditTime: 2022-09-20 23:58:21
 * @FilePath: /icode-paddle-lite/Paddle-Lite/lite/kernels/xpu/__xpu__qkv_attention_compute.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
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

#pragma once

#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

class XPUQkvAttentionCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
    using param_t = operators::XPUQkvAttentionParam;

    // void PrepareForRun() override;
    
    virtual void Run();

    virtual ~XPUQkvAttentionCompute() = default;

 private:
    // float w_max;
    // XPUScratchPadGuard q_max_guard_;
    // XPUScratchPadGuard k_max_guard_;
    // XPUScratchPadGuard v_max_guard_;
    // XPUScratchPadGuard qk_max_guard_;
    // XPUScratchPadGuard qkv_max_guard_;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
