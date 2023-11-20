import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from new_model import AlexNet
import time


start_time = time.time()
def main():
    # Set the device based on CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # perform initialization operations on the images
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0, 0, 0), (1, 1, 1))])

    # load image
    img_path = "../../images/picture_0_0_0.png.png"

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = AlexNet(num_classes=2).to(device)

    # load model weights
    weights_path = "./modified_model_deReLu_modweight.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)

    print("class: {:10}   prob: {:.3f}   raw prob: {:.3f}".format(class_indict[str(0)], predict[0].numpy(),
                                                                      output[0].numpy()))
    print("class: {:10}   prob: {:.3f}   raw prob: {:.3f}".format(class_indict[str(1)], predict[1].numpy(),
                                                                      output[1].numpy()))

    plt.show()


if __name__ == '__main__':
    main()


end_time = time.time()
execution_time = end_time - start_time
print(f"running time：{execution_time}seconds")