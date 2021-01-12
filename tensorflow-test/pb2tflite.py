import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
#in_path = '/home/weixing/Documents/models/online_used_20201202/pb'
#out_path = '/home/weixing/temp/APU_test_models/mv2_cpm_small_3_concat_stages_6_small_heat_192_128.tflite'

#in_path = '/home/weixing/temp/APU_test_models/mv2_cpm_small_3_concat_stages_6_small_heat_192_128.pb'
#out_path = '/home/weixing/temp/APU_test_models/mv2_cpm_small_3_concat_stages_6_small_heat_192_128_202101051002.tflite'

#in_path = '/home/weixing/src/30_reid/deep-person-reid/log/osnet_x0_25_market1501_softmax/osnet_x0_25_market1501_softmax.pb'
#out_path = '/home/weixing/src/30_reid/deep-person-reid/log/osnet_x0_25_market1501_softmax/osnet_x0_25_market1501_softmax.tflite'

in_path = '/home/weixing/temp/APU_test_models/osnet_x0_25_market1501_softmax_tf1154.pb'
out_path = '/home/weixing/temp/APU_test_models/osnet_x0_25_market1501_softmax_tf1154.tflite'

#in_path = '/home/weixing/temp/APU_test_models/osnet_x0_25_market1501_softmax_tf231.pb'
#in_path = '/home/weixing/temp/APU_test_models/osnet_x0_25_market1501_softmax_3.pb'
#out_path = '/home/weixing/temp/APU_test_models/osnet_x0_25_market1501_softmax_tf231.tflite'

#in_path = '/home/weixing/temp/APU_test_models/mb2-ssd-lite_0227_2.pb'
#in_path = '/home/weixing/temp/APU_test_models/mb2-ssd-lite_0227_2_tf1140.pb'
#out_path = '/home/weixing/temp/APU_test_models/mb2-ssd-lite_0227_2_tf1140.tflite'

#input_tensor_name = '0'
#class_tensor_name = ['Softmax', 'concat_88']

#input_tensor_name = 'image'
#class_tensor_name = ['Convolutional_Pose_Machine/stage_5_mv1/stage_5_mv1_2_pointwise/Relu']

input_tensor_name = 'input'
class_tensor_name = ['output'] #['Relu_397']

#converter=tf.lite.TFLiteConverter.from_saved_model(in_path)
#model = tf.saved_model.load(in_path)
#concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
#concrete_func.inputs[0].set_shape([1,3,256,128])
#converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

#converter=tf.lite.TFLiteConverter.from_frozen_graph(in_path)
#converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(in_path)
#converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(in_path, input_arrays=[input_tensor_name], output_arrays=class_tensor_name)
converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path, input_arrays=[input_tensor_name], output_arrays=class_tensor_name)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
#converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
#converter.allow_custom_ops=True
converter.experimental_new_converter =True
tflite_model=converter.convert()
 
 
with open(out_path,'wb') as f:
    f.write(tflite_model)