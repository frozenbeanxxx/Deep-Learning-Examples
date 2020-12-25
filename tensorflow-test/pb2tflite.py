import tensorflow as tf
 
#in_path = '/home/weixing/Documents/models/online_used_20201202/pb'
in_path = '/home/weixing/temp/APU_test_models/mv2_cpm_small_3_concat_stages_6_small_heat_192_128.pb'
#out_path = '/home/weixing/temp/APU_test_models/mv2_cpm_small_3_concat_stages_6_small_heat_192_128.tflite'
out_path = '/home/weixing/temp/APU_test_models/mv2_cpm_small_3_concat_stages_6_small_heat_192_128_quant2.tflite'

input_tensor_name = 'image'
class_tensor_name = 'Convolutional_Pose_Machine/stage_5_mv1/stage_5_mv1_2_pointwise/Relu'
#convertr=tf.lite.TFLiteConverter.from_frozen_graph(in_path)
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(in_path, input_arrays=[input_tensor_name], output_arrays=[class_tensor_name])
#converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path, input_arrays=[input_tensor_name], output_arrays=[class_tensor_name])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model=converter.convert()
 
 
with open(out_path,'wb') as f:
    f.write(tflite_model)