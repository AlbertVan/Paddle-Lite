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

#include "lite/operators/__xpu__qkv_attention_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUQkvAttentionOp::CheckShape() const {
  //std::cout << "==========CheckShape============= " << std::endl;
  CHECK_EQ(param_.input->dims().size(), 3UL);
  return true;
}

bool XPUQkvAttentionOp::InferShapeImpl() const {
  //std::cout << "==========InferShapeImpl============= " << std::endl;
  auto input_shape = param_.input->dims();
  auto batch_size = input_shape[0];
  auto seq_len = input_shape[1];
  // auto head_num = input_shape[2];
  int qkv_len = input_shape[2];  
  int dim = 3; // q k v
  int multi_head_size = qkv_len / dim;
  int head_num = param_.head_num;
  int size_per_head = multi_head_size / head_num;
  param_.output->Resize({batch_size, seq_len, multi_head_size});
  return true;
}

bool XPUQkvAttentionOp::AttachImpl(const cpp::OpDesc& op_desc,
                                   lite::Scope* scope) {
  //std::cout << "impl_init " << std::endl;
  param_.input = const_cast<lite::Tensor*>(
      &scope->FindVar(op_desc.Input("X").front())->Get<lite::Tensor>());
  param_.output = scope->FindVar(op_desc.Output("Out").front())
                      ->GetMutable<lite::Tensor>();
  param_.head_num = op_desc.GetAttr<int>("head_num");
  param_.size_per_head = op_desc.GetAttr<int>("size_per_head");
  param_.act_type = op_desc.GetAttr<std::string>("act_type");
  param_.precision = op_desc.GetAttr<std::string>("precision");
  param_.enable_qkv_fusion = op_desc.GetAttr<bool>("enable_qkv_fusion");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__qkv_attention,
                 paddle::lite::operators::XPUQkvAttentionOp);
