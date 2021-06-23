import onnx

from onnx_tf.backend import prepare

in_path = '/home/weixing/temp/APU_test_models/osnet_x0_25_market1501_softmax_3.onnx'
out_path = '/home/weixing/temp/APU_test_models/osnet_x0_25_market1501_softmax_3.pb'
onnx_model = onnx.load(in_path)  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph(out_path)  # export the model
