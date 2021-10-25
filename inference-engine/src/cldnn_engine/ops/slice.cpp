// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cldnn_program.h"
#include "cldnn_common_utils.h"
#include "ngraph/op/slice.hpp"
#include "cldnn/primitives/slice.hpp"
#include <memory>

namespace CLDNNPlugin {

namespace {

void CreateSliceOp(Program& p, const std::shared_ptr<ngraph::op::v8::Slice>& op) {
    p.ValidateInputs(op, { 4, 5 });
    auto input_primitives = p.GetInputPrimitiveIDs(op);
    auto output_shape = CldnnTensorFromIEDims(op->get_output_shape(0));
    auto slice_prim = cldnn::slice(layer_type_name_ID(op),
            input_primitives, output_shape,
            op->get_friendly_name());
    p.AddPrimitive(slice_prim);
    p.AddPrimitiveToProfiler(op);
}

} // namespace

REGISTER_FACTORY_IMPL(v8, Slice);

} // namespace CLDNNPlugin
