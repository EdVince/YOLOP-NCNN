## YOLOP-NCNN
将YOLOP的模型搬运到NCNN上，工程里面给了windows下的VS测试以及安卓实现
### YOLOP
YOLOP：车辆检测+路面分割+车道线分割 三合一的网络，基于YOLO系列设计的，官方的工程在这：https://github.com/hustvl/YOLOP
### 工程细节
1. VS2019上跑NCNN的工程参考的这个：https://github.com/EdVince/Ncnn-Win/tree/main/vs2019_ncnn_opencv-mobile_demo
2. 安卓上跑NCNN的工程参考的这个：https://github.com/nihui/ncnn-android-nanodet
### 实现细节
这个主要的问题在于将原YOLOP的基于pytorch的模型转换成ncnn的模型，并在ncnn上成功运行
##### 模型转换(安卓工程下给了可用的模型，不需要重新训练的可以直接用)
1. 在原YOLOP工程中，stride 8/16/32的输出是拼接了的，要注释掉这一部分的代码，并用pytorch的export导出onnx模型
2. 用onnxsim过一遍导出来的onnx模型
3. 用onnx2ncnn工具将onnx模型转成ncnn模型
4. 参考nihui的YoloV5的文章：https://zhuanlan.zhihu.com/p/275989233 ，手动修改ncnn模型的Focus的六个slices头和输出的动态reshape
5. 用ncnnoptimeze过一遍修改后的模型顺带转成fp16节省空间
##### NCNN运行
1. 参考nihui的YoloV5的实现：https://github.com/nihui/ncnn-android-yolov5 ，手动做stride 8/16/32的繁杂的后处理
### 目前问题
1. 慢
2. 为了省事，安卓的工程并没有做动态尺寸输入，是直接转成了640×640跑的代码，转成动态尺寸输入的话，能有一定的推理速度上的提升
3. 目前虽然模型是fp16的，但是这个ncnn的fp16推理我不太会弄，有懂哥可以自己试一下，看下能不能有较为明显的提速
### 安卓结果
我导出了APP，给大家下载玩玩: https://github.com/EdVince/YOLOP-NCNN/blob/main/com.tencent.nanodetncnn-debug.apk
![image](https://github.com/EdVince/YOLOP-NCNN/blob/main/res2.jpg)
