still under construction

# Deeplab3+ TensorFlow SyncBN
This is an example to show how to implement synchronized or cross-gpu batch normalization [1] with TensorFlow.
I use Deeplab3+[2] on PASCAL VOC 2012 semantic segmentation[3] as the context to test.

# TensorFlow Version
You need to use tensorflow 1.13+
if you want to use versions lower than 1.13, you need to make the following change to the import of nccl library

# Important Feature
If you are only looking for a way to implement SyncBN and do not care about deeplab or semantic segmentation, simply focus on the batch_norm function in the file of deeplab_plus_sync.py

# Why we need SyncBN
The work of [1] has focused on this topic and I will briefly illustrate their ideas here. For more details, please refer to their paper[1].
Modern deep learning framework including TensorFlow and PyTorch compute the mean and variance of batch features within the individual GPU due to efficency issue.
This would be fine when you are working on computer vision tasks like classification or when you want to train ImageNet.
Because the spatial size of the feature map in these tasks is small and you can usually fit more than 32 batches on a 12GB GPU(TITAN X, 1080 Ti).
However, for tasks like object detection or semantic segmentation, the spatial size of the feature map is crutial to achieve good performance and we can only fit less than 8 batches on a 12GB GPU.
So to avoid this issue, we need to mannually implement batch normalization.
