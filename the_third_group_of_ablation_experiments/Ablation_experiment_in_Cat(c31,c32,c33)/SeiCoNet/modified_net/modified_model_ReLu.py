#remove the ReLU layer and dropout layer in the original AlexNet network
import torch
import time

start_time = time.time()

#load weights file
path = './AlexNet.pth'
pretrained_dict = torch.load(path)

#modify the layer hierarchy correspondence
pretrained_dict["features.2.weight"] = pretrained_dict.pop("features.3.weight")
pretrained_dict["features.2.bias"] = pretrained_dict.pop("features.3.bias")

pretrained_dict["features.4.weight"] = pretrained_dict.pop("features.6.weight")
pretrained_dict["features.4.bias"] = pretrained_dict.pop("features.6.bias")

pretrained_dict["features.5.weight"] = pretrained_dict.pop("features.8.weight")
pretrained_dict["features.5.bias"] = pretrained_dict.pop("features.8.bias")

pretrained_dict["features.6.weight"] = pretrained_dict.pop("features.10.weight")
pretrained_dict["features.6.bias"] = pretrained_dict.pop("features.10.bias")

pretrained_dict["classifier.0.weight"] = pretrained_dict.pop("classifier.1.weight")
pretrained_dict["classifier.0.bias"] = pretrained_dict.pop("classifier.1.bias")

pretrained_dict["classifier.1.weight"] = pretrained_dict.pop("classifier.4.weight")
pretrained_dict["classifier.1.bias"] = pretrained_dict.pop("classifier.4.bias")

pretrained_dict["classifier.2.weight"] = pretrained_dict.pop("classifier.6.weight")
pretrained_dict["classifier.2.bias"] = pretrained_dict.pop("classifier.6.bias")

#save new weights file
torch.save(pretrained_dict, './modified_model_deReLu_time.pth')


end_time = time.time()
execution_time = end_time - start_time
print(f"running time：{execution_time}seconds")