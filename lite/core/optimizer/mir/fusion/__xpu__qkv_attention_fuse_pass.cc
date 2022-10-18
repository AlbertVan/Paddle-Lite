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

#include <memory>
#include <string>
#include "lite/backends/xpu/math.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class XPUQkvAttentionFuser : public FuseBase {
    public:
        void BuildPattern() override {
            // std::cout << "=====XPUQkvAttentionFuser====" << std::endl;
            auto* input = 
                VarNode("input")->assert_is_op_input("reshape2", "X")
                                ->assert_only_one_output()
                                ->AsInput();

            // auto f_reshape2_attr = [](const Node* node) -> bool {
            //     auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
            //     auto input_y_name = op_desc.Input("X").front();
            //     auto* scope = const_cast<Node*>(node)->AsStmt().op()->scope();
            //     auto y_shape = scope->FindVar(input_y_name)->Get<lite::Tensor>().dims();
            //     size_t y_rank = y_shape.size();
            //     std::cout << "y_rank:" << y_rank <<std::endl;
            //     return (y_rank == 3);
            //     };
            auto* f_reshape2 = OpNode("f_reshape2", "reshape2")
                                //->assert_node_satisfied(f_reshape2_attr)
                                ->AsIntermediate();

            auto* f_reshape2_out = 
                VarNode("f_reshape2_out")->assert_is_op_output("reshape2", "Out")
                                ->assert_is_op_input("transpose2", "X")
                                ->AsIntermediate();
            auto* f_reshape2_xshape = 
                VarNode("f_reshape2_xshape")
                                ->assert_is_op_output("reshape2", "XShape")
                                ->AsIntermediate();

            auto* f_transpose2 = 
                OpNode("f_transpose2", "transpose2")
                                ->assert_op_attr<std::vector<int32_t>>("axis", {2, 0, 3, 1, 4})
                                ->AsIntermediate();

            auto* f_transpose2_out = 
                VarNode("f_transpose2_out")
                                ->assert_is_op_output("transpose2", "Out")
                                ->assert_is_op_input("slice", "Input")
                                ->AsIntermediate();

            auto* f_transpose2_xshape = 
                VarNode("f_transpose2_xshape")
                                ->assert_is_op_output("transpose2", "XShape")
                                ->AsIntermediate();
            
            auto* slice0 = 
                OpNode("slice0", "slice")
                                ->assert_op_attr_satisfied<std::vector<int>>(
                                    "axes",
                                    [](const std::vector<int>& attr) {
                                    return attr.size() == 1 && attr[0] == 0;
                                    })
                                ->assert_op_attr_satisfied<std::vector<int>>(
                                    "starts",
                                    [](const std::vector<int>& attr) { return attr.size() == 1 && attr[0] == 0;})
                                ->assert_op_attr_satisfied<std::vector<int>>(
                                    "ends",
                                    [](const std::vector<int>& attr) { return attr.size() == 1 && attr[0] == 1;})
                                ->AsIntermediate();

            auto* slice0_out = 
                VarNode("slice0_out")->assert_is_op_output("slice", "Out")
                                     ->assert_is_op_input("matmul_v2", "X")
                                     ->AsIntermediate();
            
            auto* slice1 = 
                OpNode("slice1", "slice")
                                    ->assert_op_attr_satisfied<std::vector<int>>(
                                        "axes",
                                        [](const std::vector<int>& attr) {
                                        return attr.size() == 1 && attr[0] == 0;
                                        })
                                    ->assert_op_attr_satisfied<std::vector<int>>(
                                        "starts",
                                        [](const std::vector<int>& attr) { return attr.size() == 1 && attr[0] == 1;})
                                    ->assert_op_attr_satisfied<std::vector<int>>(
                                        "ends",
                                        [](const std::vector<int>& attr) { return attr.size() == 1 && attr[0] == 2;})
                                    ->AsIntermediate();
            
            auto* slice1_out = 
                VarNode("slice1_out")
                                ->assert_is_op_output("slice", "Out")
                                ->assert_is_op_input("transpose2", "X")
                                ->AsIntermediate();

            auto* slice2 = 
                OpNode("slice2", "slice")
                                    ->assert_op_attr_satisfied<std::vector<int>>(
                                        "axes",
                                        [](const std::vector<int>& attr) {
                                        return attr.size() == 1 && attr[0] == 0;
                                        })
                                    ->assert_op_attr_satisfied<std::vector<int>>(
                                        "starts",
                                        [](const std::vector<int>& attr) { return attr.size() == 1 && attr[0] == 2;})
                                    ->assert_op_attr_satisfied<std::vector<int>>(
                                        "ends",
                                        [](const std::vector<int>& attr) { return attr.size() == 1 && attr[0] == 3;})
                                    ->AsIntermediate();
            
            auto* slice2_out = 
                VarNode("slice2_out")->assert_is_op_output("slice", "Out")
                                ->assert_is_op_input("matmul_v2", "Y")
                                ->AsIntermediate();

            auto* s_transpose2 = 
                OpNode("s_transpose2", "transpose2")
                            ->assert_op_attr<std::vector<int32_t>>("axis", {0, 1, 3, 2})
                            ->AsIntermediate();
            
            auto* s_transpose2_xshape = 
                VarNode("s_transpose2_xshape")
                            ->assert_is_op_output("transpose2", "XShape")
                            ->AsIntermediate();

            auto* s_transpose2_out = 
                VarNode("s_transpose2_out")->assert_is_op_output("transpose2", "Out")
                                ->assert_is_op_input("matmul_v2", "Y")
                                ->AsIntermediate();

            auto* f_matmul = 
                OpNode("f_matmul", "matmul_v2")->AsIntermediate();

            auto* f_matmul_out = 
                VarNode("f_matmul_out")->assert_is_op_output("matmul_v2", "Out")
                                ->assert_is_op_input("scale", "X")
                                ->AsIntermediate();

            auto* scale = 
                OpNode("scale", "scale")
                                //->assert_op_attr<float>("bias", 0)
                                ->AsIntermediate();
            
            auto* scale_out = 
                 VarNode("scale_out")
                                ->assert_is_op_output("scale", "Out")
                                ->assert_is_op_input("softmax", "X")
                                ->AsIntermediate();

            auto* softmax = 
                OpNode("softmax", "softmax")
                                //->assert_op_attr<int32_t>("axis", -1)
                                ->AsIntermediate();
            
            auto* softmax_out = 
                 VarNode("softmax_out")
                                ->assert_is_op_output("softmax", "Out")
                                ->assert_is_op_input("matmul_v2", "X")
                                ->AsIntermediate();

            auto* s_matmul = 
                OpNode("s_matmul", "matmul_v2")->AsIntermediate();

            auto* s_matmul_out = VarNode("s_matmul_out")
                                ->assert_is_op_output("matmul_v2", "Out")
                                ->assert_is_op_input("transpose2", "X")
                                ->AsIntermediate();

            auto* t_transpose2 = 
                OpNode("t_transpose2", "transpose2")
                                ->assert_op_attr<std::vector<int32_t>>("axis", {0, 2, 1, 3})
                                ->AsIntermediate();

            auto* t_transpose2_out = VarNode("t_transpose2_out")
                                ->assert_is_op_output("transpose2", "Out")
                                ->assert_is_op_input("reshape2", "X")
                                ->AsIntermediate();   

            auto* t_transpose2_xshape = VarNode("t_transpose2_xshape")
                                ->assert_is_op_output("transpose2", "XShape")
                                ->AsIntermediate();   
            
            auto* s_reshape2 = 
                OpNode("s_reshape2", "reshape2")->AsIntermediate();

            auto* s_reshape2_xshape = VarNode("s_reshape2_xshape")
                                ->assert_is_op_output("reshape2", "XShape")
                                ->AsOutput();  

            auto* s_reshape2_out = VarNode("s_reshape2_out")
                                ->assert_is_op_output("reshape2", "Out")
                                ->AsOutput();                    
            
            *input >> *f_reshape2 >> *f_reshape2_out >> *f_transpose2 >> *f_transpose2_out;
            *f_transpose2_out >> *slice0 >> *slice0_out >> *f_matmul;
            *f_transpose2_out >> *slice1 >> *slice1_out >> *s_transpose2 >> *s_transpose2_out >> *f_matmul;
            *f_transpose2_out >> *slice2 >> *slice2_out >> *s_matmul;
            *f_matmul >> *f_matmul_out >> *scale >> *scale_out >> *softmax >> *softmax_out >> *s_matmul;
            *s_matmul >> *s_matmul_out >> *t_transpose2 >> *t_transpose2_out >> *s_reshape2 >> *s_reshape2_out;
            *f_reshape2 >> *f_reshape2_xshape;
            *f_transpose2 >> *f_transpose2_xshape;
            *s_transpose2 >> *s_transpose2_xshape;
            *t_transpose2 >> *t_transpose2_xshape;
            *s_reshape2 >> *s_reshape2_xshape;
  }

    void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
            cpp::OpDesc op_desc;
            op_desc.SetType("__xpu__qkv_attention");
            op_desc.SetInput("X", {matched.at("input")->arg()->name});
            op_desc.SetOutput("Out", {matched.at("s_reshape2_out")->arg()->name});
            auto shape_info = *matched.at("f_reshape2")->stmt()->op_info();
            auto input = shape_info.inputs();
            std::vector<int> f_shape_info = {shape_info.GetAttr<std::vector<int>>("shape")[0],
                                            shape_info.GetAttr<std::vector<int>>("shape")[1],
                                            shape_info.GetAttr<std::vector<int>>("shape")[2],
                                            shape_info.GetAttr<std::vector<int>>("shape")[3],
                                            shape_info.GetAttr<std::vector<int>>("shape")[4]};
            // for (size_t i = 0; i < f_shape_info.size(); i++){
            //     std::cout << f_shape_info[i] << std::endl;
            // }
            // int batch_size = f_shape_info[0];
            int head_num = f_shape_info[3];
            int size_per_head = f_shape_info[4];
            std::string precision = "int31";

            op_desc.SetAttr<std::string>("precision", precision);
            op_desc.SetAttr<int>("head_num", head_num);
            op_desc.SetAttr<int>("size_per_head", size_per_head);
            op_desc.SetAttr<bool>("enable_qkv_fusion", true);
            op_desc.SetAttr<std::string>("act_type", "relu");
            
            auto reshape2 = matched.at("s_reshape2")->stmt()->op();
            auto* scope = reshape2->scope();
            auto& valid_places = reshape2->valid_places();
            // std::cout << op_desc.Type() << std::endl;
            auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
            new_op->Attach(op_desc, scope);
            auto* new_op_node = graph->GraphCreateInstructNode(new_op, valid_places);
            CHECK(new_op_node != nullptr) << " GraphCreateInstructNode failed";

            IR_NODE_LINK_TO(matched.at("input"), new_op_node);
            IR_NODE_LINK_TO(new_op_node, matched.at("s_reshape2_out"));
            //std::cout << "=====XPUQkvAttentionFuser_TEST5====" << std::endl;
    }
};

}  // namespace fusion

class XPUQkvAttentionFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPUQkvAttentionFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__qkv_attention_fuse_pass,
                  paddle::lite::mir::XPUQkvAttentionFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__qkv_attention");

