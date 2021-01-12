import torch

myTensor = torch.randn(2, 2,requires_grad=True)
helper_tensor = torch.ones(2, 2)
new_myTensor = myTensor * helper_tensor # new tensor, out-of-place operation
new_myTensor[0, 0] = 50
with torch.enable_grad():
    x=new_myTensor.sum() *10 # of course you need to use the new tensor
x.backward()                 # for further calculation and backward
print(myTensor.grad)