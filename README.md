
<p align="center">
  <img width="1200" src="/assets/banner.PNG">
</p>

# automatic-mixed-precision-tutorials-pytorch
Automatic Mixed Precision Tutorials using pytorch. Based on [PyTorch 1.6 Official Features (Automatic Mixed Precision)](https://pytorch.org/docs/stable/notes/amp_examples.html), implement classification codebase using custom dataset.

- author: hoya012  
- last update: 2020.08.24
- [supplementary materials (blog post written in Korean)](https://hoya012.github.io/blog/Mixed-Precision-Training/)

## 0. Experimental Setup (I used 1 GTX 1080 Ti GPU!)
### 0-1. Prepare Library
- Must use Newest PyTorch version. (>= 1.6.0)

```python
pip install -r requirements.txt
```

### 0-2. Download dataset (Kaggle Intel Image Classification)

- [Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification/)

This Data contains around 25k images of size 150x150 distributed under 6 categories.
{'buildings' -> 0,
'forest' -> 1,
'glacier' -> 2,
'mountain' -> 3,
'sea' -> 4,
'street' -> 5 }

### 1. Baseline Training 
- ImageNet Pretrained ResNet-18 from torchvision.models
- Batch Size 256 / Epochs 120 / Initial Learning Rate 0.0001
- Training Augmentation: Resize((256, 256)), RandomHorizontalFlip()
- Adam + Cosine Learning rate schedueling with warmup
- I tried NVIDIA Pascal GPU - GTX 1080 Ti 1 GPU (w/o Tensor Core) and NVIDIA Turing GPU - RTX 2080 Ti 1 GPU (with Tensor Core)

```python
python main.py --checkpoint_name baseline;
```

### 2. Automatic Mixed Precision Training 

In PyTorch 1.6, Automatic Mixed Precision Training is very easy to use! Thanks to PyTorch..

```python
""" define loss scaler for automatic mixed precision """
scaler = torch.cuda.amp.GradScaler()

for batch_idx, (inputs, labels) in enumerate(data_loader):
  self.optimizer.zero_grad()

  with torch.cuda.amp.autocast():
    outputs = self.model(inputs)
    loss = self.criterion(outputs, labels)

  # Scales the loss, and calls backward() 
  # to create scaled gradients 
  self.scaler.scale(loss).backward()

  # Unscales gradients and calls 
  # or skips optimizer.step() 
  self.scaler.step(self.optimizer)

  # Updates the scale for next iteration 
  self.scaler.update()
```

#### Run Script (Command Line)
```python
python main.py --checkpoint_name baseline_amp --amp;
```

### 3. Performance Table
- B : Baseline (FP32)
- AMP : Automatic Mixed Precision Training (AMP)

|   Algorithm  | Test Accuracy |   GPU Memory   | Total Training Time |
|:------------:|:-------------:|:--------------:|:-------------------:|
|  B - 1080 Ti |      94.13    |     10737MB    |         64.9m       |    
|  B - 2080 Ti |      94.17    |     10855MB    |         54.3m       |    
| AMP - 1080 Ti|      94.07    |     6615MB     |         64.7m       |  
| AMP - 2080 Ti|      94.23    |     7799MB     |         37.3m       |  

### 4. Code Reference
- Baseline Code: https://github.com/hoya012/carrier-of-tricks-for-classification-pytorch
- Gradual Warmup Scheduler: https://github.com/ildoonet/pytorch-gradual-warmup-lr
- PyTorch Automatic Mixed Precision: https://pytorch.org/docs/stable/notes/amp_examples.html