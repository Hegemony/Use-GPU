# Use-GPU


## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0+](http://pytorch.org/)


### 1.计算设备

PyTorch可以指定用来存储和计算的设备，如使用内存的CPU或者使用显存的GPU。默认情况下，PyTorch会将数据创建在内存，然后利用CPU来计算。

用torch.cuda.is_available()查看GPU是否可用:

### 2.Tensor的GPU计算

默认情况下，Tensor会被存在内存上。因此，之前我们每次打印Tensor的时候看不到GPU相关标识。

使用.cuda()可以将CPU上的Tensor转换（复制）到GPU上。如果有多块GPU，我们用.cuda(i)来表示第 ii 块GPU及相应的显存（ii从0开始）且cuda(0)和cuda()等价。

我们可以通过Tensor的device属性来查看该Tensor所在的设备。

### 3.模型的GPU计算

同Tensor类似，PyTorch模型也可以通过.cuda转换到GPU上。我们可以通过检查模型的参数的device属性来查看存放模型的设备。

### PyTorch可以指定用来存储和计算的设备，如使用内存的CPU或者使用显存的GPU。在默认情况下，PyTorch会将数据创建在内存，然后利用CPU来计算。
### PyTorch要求计算的所有输入数据都在内存或同一块显卡的显存上。
