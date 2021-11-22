// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <tuple>
#include <cctype>
#include <memory>

#include "cldnn_transformations_pipeline.h"

#include "ie_metric_helpers.hpp"
#include "ie_plugin_config.hpp"
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ie_ngraph_utils.hpp>
#include <ie_algorithm.hpp>

#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>

#include <transformations/control_flow/unroll_tensor_iterator.hpp>

#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"
#include <transformations/op_conversions/convert_depth_to_space.hpp>
#include <transformations/op_conversions/convert_space_to_depth.hpp>
#include <transformations/op_conversions/convert_gelu.hpp>
#include <transformations/op_conversions/convert_mod.hpp>
#include <transformations/op_conversions/convert_broadcast3.hpp>
#include <transformations/op_conversions/reduce_l1_decomposition.hpp>
#include <transformations/op_conversions/reduce_l2_decomposition.hpp>
#include <transformations/op_conversions/convert_pad_to_group_conv.hpp>
#include <transformations/op_conversions/softplus_decomposition.hpp>
#include <transformations/op_conversions/convert_space_to_batch.hpp>
#include <transformations/op_conversions/convert_batch_to_space.hpp>
#include <transformations/op_conversions/convert_reduce_to_pooling.hpp>
#include <transformations/op_conversions/convert_shuffle_channels3.hpp>
#include <transformations/op_conversions/hswish_decomposition.hpp>
#include <transformations/op_conversions/hsigmoid_decomposition.hpp>
#include <transformations/op_conversions/log_softmax_decomposition.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/op_conversions/convert_subtract.hpp>
#include <transformations/op_conversions/convert_ti_to_sequences.hpp>
#include <transformations/op_conversions/gru_cell_decomposition.hpp>
#include <transformations/op_conversions/lstm_cell_decomposition.hpp>
#include <transformations/op_conversions/rnn_cell_decomposition.hpp>
#include <transformations/op_conversions/mvn6_decomposition.hpp>
#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_previous_nms_to_nms_5.hpp>
#include <transformations/op_conversions/convert_nms_to_nms_ie_internal.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>
#include <transformations/op_conversions/convert_gather_downgrade.hpp>
#include <transformations/op_conversions/convert_gather_0d.hpp>
#include <transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp>
#include <transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp>
#include "transformations/op_conversions/softmax_decomposition.hpp"
#include "transformations/op_conversions/convert_slice_to_strided_slice.hpp"
#include <transformations/convert_precision.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>

#include <transformations/low_precision/disable_convert_constant_folding_on_const_path.hpp>
#include <low_precision/pull_reshape_through_dequantization.hpp>
#include <low_precision/pull_transpose_through_dequantization.hpp>
#include <low_precision/convolution.hpp>
#include <low_precision/convolution_backprop_data.hpp>
#include <low_precision/group_convolution.hpp>
#include <low_precision/low_precision.hpp>
#include <low_precision/mat_mul.hpp>
#include <low_precision/multiply_to_group_convolution.hpp>
#include <low_precision/strided_slice.hpp>
#include <low_precision/network_helper.hpp>

#include "cldnn_itt.h"

namespace {
template<typename T>
static bool disableReduceDecomposition(const std::shared_ptr<const ngraph::Node> node) {
    if (auto op = std::dynamic_pointer_cast<const T>(node)) {
        bool fp16_batch_not_1 = op->get_element_type() == ngraph::element::f16 && op->input(0).get_shape()[0] != 1;
        return !fp16_batch_not_1;
    }
    return false;
}
}  // namespace

namespace CLDNNPlugin {

void TransformationsPipeline::apply(std::shared_ptr<ov::Function> func) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "TransformationsPipeline::apply");
    using const_node_ptr = const std::shared_ptr<const ngraph::Node>;

    bool use_onednn = false;
#ifdef ENABLE_ONEDNN_FOR_GPU
    use_onednn = device_info.supports_immad;
#endif

    bool enableInt8;
    {
        ngraph::pass::Manager manager;
        manager.set_per_pass_validation(false);

        enableInt8 = config.enableInt8 && ngraph::pass::low_precision::LowPrecision::isFunctionQuantized(func);
        if (enableInt8) {
            manager.register_pass<ngraph::pass::DisableConvertConstantFoldingOnConstPath>(
                std::vector<ngraph::element::Type>{ ngraph::element::i8, ngraph::element::u8, ngraph::element::i4, ngraph::element::u4 });
        }

        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::CommonOptimizations>();

        if (!config.enable_loop_unrolling) {
            manager.register_pass<ngraph::pass::BidirectionalLSTMSequenceDecomposition>();
            manager.register_pass<ngraph::pass::BidirectionalGRUSequenceDecomposition>();
            manager.register_pass<ngraph::pass::BidirectionalRNNSequenceDecomposition>();
        }

        manager.register_pass<ngraph::pass::ConvertRNNSequenceToTensorIterator>();
        manager.register_pass<ngraph::pass::ConvertGRUSequenceToTensorIterator>();
        manager.register_pass<ngraph::pass::ConvertLSTMSequenceToTensorIterator>();
        manager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
        manager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();

        manager.register_pass<ngraph::pass::ConvertTensorIteratorToGRUSequence>();
        manager.register_pass<ngraph::pass::ConvertTensorIteratorToLSTMSequence>();
        manager.register_pass<ngraph::pass::ConvertTensorIteratorToRNNSequence>();
        manager.register_pass<ngraph::pass::LSTMCellDecomposition>();
        manager.register_pass<ngraph::pass::GRUCellDecomposition>();
        manager.register_pass<ngraph::pass::RNNCellDecomposition>();

        if (config.enable_loop_unrolling) {
            manager.register_pass<ngraph::pass::BidirectionalLSTMSequenceDecomposition>();
            manager.register_pass<ngraph::pass::BidirectionalGRUSequenceDecomposition>();
            manager.register_pass<ngraph::pass::BidirectionalRNNSequenceDecomposition>();
        }

        manager.register_pass<ngraph::pass::ConvertNMS1ToNMS5>();
        manager.register_pass<ngraph::pass::ConvertNMS3ToNMS5>();
        manager.register_pass<ngraph::pass::ConvertNMS4ToNMS5>();
        manager.register_pass<ngraph::pass::ConvertNMSToNMSIEInternal>();
        manager.register_pass<ngraph::pass::ConvertGather0D>();

        static const precisions_array convert_precision_list {
                {ngraph::element::i64, ngraph::element::i32},
                {ngraph::element::u64, ngraph::element::i32},
                {ngraph::element::u16, ngraph::element::i32},
                {ngraph::element::u32, ngraph::element::i32},
                {ngraph::element::boolean, ngraph::element::u8},
                {ngraph::element::i4, ngraph::element::i8},
                {ngraph::element::u4, ngraph::element::u8},
        };

        manager.register_pass<ngraph::pass::Validate>();
        manager.register_pass<ngraph::pass::ConvertPrecision>(convert_precision_list);

        auto pass_config = manager.get_pass_config();

        pass_config->enable<ov::pass::ConvertCompressedOnlyToLegacy>();

        // SpaceToDepth/DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
        pass_config->set_callback<ngraph::pass::ConvertSpaceToDepth,
                                  ngraph::pass::ConvertDepthToSpace>(
                [](const_node_ptr &node) -> bool {
                    return node->input_value(0).get_shape().size() <= 5lu &&
                        node->input_value(0).get_shape().size() == node->get_output_shape(0).size();
                });

        pass_config->set_callback<ngraph::pass::ConvertBatchToSpace,
                                  ngraph::pass::ConvertSpaceToBatch>(
                [](const_node_ptr &node) -> bool {
                    const auto & rank = node->input(0).get_partial_shape().rank().get_length();
                    return rank <= 5lu;
                });

        pass_config->set_callback<ngraph::pass::ConvertReduceSumToPooling>(
            [](const_node_ptr &node) -> bool {
                return disableReduceDecomposition<ngraph::opset1::ReduceSum>(node);
            });

        pass_config->set_callback<ngraph::pass::ConvertReduceMeanToPooling>(
            [](const_node_ptr &node) -> bool {
                return disableReduceDecomposition<ngraph::opset1::ReduceMean>(node);
            });

        pass_config->set_callback<ngraph::pass::ConvertReduceMaxToPooling>(
            [](const_node_ptr &node) -> bool {
                return disableReduceDecomposition<ngraph::opset1::ReduceMax>(node);
            });

        auto isCellPrimitiveSupported = [](const_node_ptr &node) -> bool {
            if (std::dynamic_pointer_cast<const ngraph::opset6::RNNCell>(node)) {
                return false;
            } else if (std::dynamic_pointer_cast<const ngraph::opset6::GRUCell>(node)) {
                return false;
            } else if (const auto &lstm_cell = std::dynamic_pointer_cast<const ngraph::opset6::LSTMCell>(node)) {
                return lstm_cell->get_clip() == 0.0f && lstm_cell->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
            } else if (const auto &lstm_cell_v1 = std::dynamic_pointer_cast<const ngraph::opset1::LSTMCell>(node)) {
                return lstm_cell_v1->get_clip() == 0.0f && lstm_cell_v1->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
            }
            return false;
        };

        // Sequences supported by the plugin shouldn't be converted to TensorIterator.
        // sequence_length input is not supported in all Sequences, so if is_seq_len_provided() == true, we
        // should always convert to TensorIterator.
        // RNN/GRU Sequences are not supported in GPU plugin
        // LSTM Sequence supported with clip == 0, and activations have default values (sigmoid, tanh, tanh)
        auto isSequencePrimitiveSupported = [](const_node_ptr &node) -> bool {
            const auto& data = node->input(0);
            const auto& data_pshape = data.get_partial_shape();
            if (data_pshape.rank().is_static() && data_pshape.rank().get_length() > 1 && !data_pshape[1].is_static())
                return false;
            auto max_seq_len = data.get_shape().at(1);
            if (std::dynamic_pointer_cast<const ngraph::opset6::RNNSequence>(node)) {
                return false;
            } else if (std::dynamic_pointer_cast<const ngraph::opset6::GRUSequence>(node)) {
                return false;
            } else if (const auto &lstm_seq = std::dynamic_pointer_cast<const ngraph::opset6::LSTMSequence>(node)) {
                return lstm_seq->get_clip() == 0.0f &&
                       lstm_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"} &&
                       !ngraph::op::util::is_seq_len_provided(lstm_seq->get_input_node_shared_ptr(3),
                                                              max_seq_len);
            }
            return false;
        };

        pass_config->set_callback<ngraph::pass::RNNCellDecomposition,
                                  ngraph::pass::GRUCellDecomposition,
                                  ngraph::pass::LSTMCellDecomposition>(
            [isCellPrimitiveSupported](const_node_ptr &node) -> bool {
                return isCellPrimitiveSupported(node);
            });

        pass_config->set_callback<ngraph::pass::ConvertRNNSequenceToTensorIterator,
                                  ngraph::pass::ConvertGRUSequenceToTensorIterator,
                                  ngraph::pass::ConvertLSTMSequenceToTensorIterator>(
                [isSequencePrimitiveSupported](const_node_ptr &node) -> bool {
                    return isSequencePrimitiveSupported(node);
                });

        pass_config->set_callback<ngraph::pass::ConvertTensorIteratorToRNNSequence,
                                  ngraph::pass::ConvertTensorIteratorToLSTMSequence,
                                  ngraph::pass::ConvertTensorIteratorToGRUSequence>(
            [isCellPrimitiveSupported](const_node_ptr &node) -> bool {
                if (const auto& ti_op = std::dynamic_pointer_cast<const ngraph::op::TensorIterator>(node)) {
                    size_t count_rnn = 0;
                    for (const auto &op : ti_op->get_body()->get_ops())
                        count_rnn += isCellPrimitiveSupported(op);
                    return count_rnn != 1;
                }
                return true;
            });

        pass_config->set_callback<ngraph::pass::MVN6Decomposition>(
            [](const_node_ptr &node) -> bool {
                const auto mvn = std::dynamic_pointer_cast<const ngraph::op::v6::MVN>(node);
                if (mvn != nullptr && node->get_input_size() == 2) {
                    if (auto axesNode = dynamic_cast<ngraph::op::v0::Constant*>(mvn->get_input_node_ptr(1))) {
                        auto axesVal = axesNode->cast_vector<int>();
                        auto& mvnShape = mvn->get_output_shape(0);
                        for (int32_t& axis : axesVal)
                            axis = axis < 0 ? axis + mvnShape.size() : axis;
                        std::sort(axesVal.begin(), axesVal.end());
                        if (mvnShape.size() == 1)
                            return false;
                        if (mvnShape.size() > 5 || (mvnShape.size() != axesVal.size() + 1 && mvnShape.size() != axesVal.size() + 2))
                            return false;
                        int value = mvnShape.size() - 1;
                        for (int i = axesVal.size() - 1; i >= 0; i--, value--) {
                            if (axesVal[i] != value)
                                return false;
                        }
                        return true;
                    }
                }
                return false;
            });

        pass_config->enable<ngraph::pass::SoftmaxDecomposition>();
        pass_config->set_callback<ngraph::pass::SoftmaxDecomposition>(
            [](const_node_ptr &node) -> bool {
                return node->input_value(0).get_partial_shape().rank().get_length() <= 5;
            });

        // List of enabled/disabled transformations
        pass_config->disable<ngraph::pass::ConvertGELU>();
        pass_config->disable<ngraph::pass::ConvertMod>();
        pass_config->disable<ngraph::pass::ConvertShuffleChannels3>();
        pass_config->disable<ngraph::pass::HSwishDecomposition>();
        pass_config->disable<ngraph::pass::HSigmoidDecomposition>();
        pass_config->disable<ngraph::pass::ReduceL1Decomposition>();
        pass_config->disable<ngraph::pass::ReduceL2Decomposition>();
        pass_config->disable<ngraph::pass::SoftPlusDecomposition>();
        pass_config->disable<ngraph::pass::LogSoftmaxDecomposition>();
        pass_config->disable<ngraph::pass::ConvertBroadcast3>();
        pass_config->disable<ngraph::pass::WeightsDequantizeToFakeQuantize>();
        pass_config->disable<ngraph::pass::SimplifyCTCGreedyDecoderSeqLen>();
        pass_config->enable<ngraph::pass::ConvertGather8ToGather7>();
        pass_config->disable<ngraph::pass::SliceToStridedSlice>();

        if (!config.enable_loop_unrolling) {
            pass_config->disable<ngraph::pass::ConvertTensorIteratorToRNNSequence>();
            pass_config->disable<ngraph::pass::ConvertTensorIteratorToLSTMSequence>();
            pass_config->disable<ngraph::pass::ConvertTensorIteratorToGRUSequence>();
        }

        pass_config->enable<ngraph::pass::ConvertInterpolate1ToInterpolate4>();

        if (enableInt8) {
            pass_config->set_callback<ngraph::pass::ConvertQuantizeDequantize>([](const_node_ptr &node) -> bool {
                return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node);
            });

            pass_config->set_callback<ngraph::pass::ConvertSubtract>([](const_node_ptr &node) -> bool {
                return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForSubtract(node);
            });
        }

        manager.run_passes(func);
    }

    if (enableInt8) {
        OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "TransformationsPipeline::apply::lpt");
        using namespace ngraph::pass::low_precision;

        // Conversion to FP32 might be needed for quantized models that face any fp16 related issues (e.g. overflow) for non-quantized layers
        // With this key users can work-around such issues
        if (!config.enable_fp16_for_quantized_models || use_onednn) {
            ngraph::pass::Manager manager;
            manager.register_pass<ngraph::pass::ConvertPrecision>(precisions_array {{ ngraph::element::f16, ngraph::element::f32 }});
            manager.run_passes(func);
        }

        auto supportedPrecisions = std::vector<OperationPrecisionRestriction>({
            OperationPrecisionRestriction::create<ngraph::opset1::Convolution>({
                {0, {ngraph::element::u8, ngraph::element::i8}},
                {1, {ngraph::element::i8}},
            }),
            OperationPrecisionRestriction::create<ngraph::opset1::ConvolutionBackpropData>({
                {0, {ngraph::element::u8, ngraph::element::i8}},
                {1, {ngraph::element::i8}}
            }),
            OperationPrecisionRestriction::create<ngraph::opset1::GroupConvolution>({
                {0, {ngraph::element::u8, ngraph::element::i8}},
                {1, {ngraph::element::i8}}
            }),
            OperationPrecisionRestriction::create<ngraph::opset1::StridedSlice>({})
        });

        auto perTensorQuantization = std::vector<OperationPerTensorQuantizationRestriction>({
            OperationPerTensorQuantizationRestriction::create<ngraph::opset1::Convolution>({0}),
            OperationPerTensorQuantizationRestriction::create<ngraph::opset1::ConvolutionBackpropData>({0}),
        });

        ngraph::pass::Manager lptManager;

        auto lptPassConfig = lptManager.get_pass_config();
        lptPassConfig->disable<ngraph::pass::low_precision::StridedSliceTransformation>();
        lptPassConfig->set_callback<ngraph::pass::low_precision::MarkupPrecisions>([](const_node_ptr& node) -> bool {
            if (const auto mulitply = std::dynamic_pointer_cast<const ngraph::opset1::Multiply>(node)) {
                return !MultiplyToGroupConvolutionTransformation::canBeTransformedToGroupConvolution(mulitply);
            }
            return false;
        });
        lptPassConfig->set_callback<ConvolutionBackpropDataTransformation>([](const_node_ptr& node) -> bool {
            auto fillStaticChannel = [](const ngraph::PartialShape& shape, size_t& channel) -> bool {
                const auto rank = shape.rank();
                if (rank.is_dynamic()) {
                    return false;
                }
                if (rank.get_length() < 2ul) {
                    return false;
                }
                const auto dimension = shape[1];
                if (dimension.is_dynamic()) {
                    return false;
                }
                channel = dimension.get_length();
                return true;
            };

            size_t inputChannels;
            if (!fillStaticChannel(node->get_input_partial_shape(0), inputChannels)) {
                return true;
            }

            size_t outputChannels;
            if (!fillStaticChannel(node->get_output_partial_shape(0), outputChannels)) {
                return true;
            }


            if ((inputChannels % 4 != 0) || (outputChannels % 16 != 0)) {
                return true;
            }

            return LayerTransformation::isAsymmetricQuantization(node) || WeightableLayerTransformation::isAsymmetricOnWeights(node);
        });
        if (!use_onednn) {
            lptPassConfig->set_callback<MatMulTransformation>([](const_node_ptr& node) -> bool {
                return MatMulTransformation::is3DTensorOnActivations(node);
            });
        }

        lptManager.register_pass<LowPrecision>(supportedPrecisions, perTensorQuantization);
        lptManager.run_passes(func);
    }

    {
        OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "TransformationsPipeline::apply::run_passes");
        ngraph::pass::Manager manager;
        // This ConstantFolding pass is added to fold reshapes added for constant inputs on NMS internal operation which prevents upper-bound calculation
        // TODO: check why we have these reshapes
        manager.register_pass<ngraph::pass::ConstantFolding>();

        manager.register_pass<ngraph::pass::UnrollTensorIterator>();
        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::UnrollTensorIterator>(
            [this](const std::shared_ptr<const ngraph::Node> &node) -> bool {
                auto sub_graph_op = std::dynamic_pointer_cast<const ngraph::op::util::SubGraphOp>(node);
                int64_t num_iter = sub_graph_op->get_num_iterations();
                if (num_iter == 1) {
                    return false;
                }
                return !config.enable_loop_unrolling;
            });

        manager.run_passes(func);
    }
}
}  // namespace CLDNNPlugin
