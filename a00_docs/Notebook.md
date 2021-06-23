## mmpose使用笔记
- mmcv的版本一定要和cuda版本、pytorch版本对应上，否则会出现奇怪的错误。我一开始没对应，在计算flops时，convtranspose2d的计算量为0的错误，辛苦发现的早，如果在训练时出错，又不报出来，那根本发现不了。