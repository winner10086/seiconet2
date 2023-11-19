import torch
from new_model import AlexNet

# Create model
model = AlexNet(num_classes = 2)

# Load model weights
model.load_state_dict(torch.load('modified_model_deReLu.pth'))

with torch.no_grad():
    for name, param in model.named_parameters():
        # Modify the parameters of the convolutional layers
        if 'features.0.weight' in name:
            kernel_params = param.clone()
            num_kernels, num_channels, kernel_size, kernel_size = kernel_params.size()

            for i in range(num_kernels):
                for j in range(num_channels):
                    kernel = kernel_params[i, j]  # Retrieve the parameters of the j-th channel of the i-th convolutional kernel
                    pos_indices = kernel > 0  # indices greater than 0
                    neg_indices = kernel < 0  # indices less than 0
                    if pos_indices.any() or neg_indices.any():
                        pos_sum = kernel[pos_indices].sum()  # sum of positive numbers
                        neg_sum = kernel[neg_indices].abs().sum()  # sum of the absolute values of negative numbers
                        k = pos_sum / (pos_sum + neg_sum)  # coefficient k
                        kernel[pos_indices] *= k  # Multiplying a positive number by a coefficient k
                        kernel[neg_indices] /= 10000  # Dividing a negative number by 10,000

            param.copy_(kernel_params)  # Assign the processed kernel_params back to the parameter param



        if 'features.2.weight' in name:
            kernel_params = param.clone()
            num_kernels, num_channels, kernel_size, kernel_size = kernel_params.size()

            for i in range(num_kernels):
                for j in range(num_channels):
                    kernel = kernel_params[i, j]
                    pos_indices = kernel > 0
                    neg_indices = kernel < 0
                    if pos_indices.any() or neg_indices.any():
                        pos_sum = kernel[pos_indices].sum()
                        neg_sum = kernel[neg_indices].abs().sum()
                        k = pos_sum / (pos_sum + neg_sum)
                        kernel[pos_indices] *= k
                        kernel[neg_indices] /= 10000

            param.copy_(kernel_params)

        if 'features.4.weight' in name:
            kernel_params = param.clone()
            num_kernels, num_channels, kernel_size, kernel_size = kernel_params.size()

            for i in range(num_kernels):
                for j in range(num_channels):
                    kernel = kernel_params[i, j]
                    pos_indices = kernel > 0
                    neg_indices = kernel < 0
                    if pos_indices.any() or neg_indices.any():
                        pos_sum = kernel[pos_indices].sum()
                        neg_sum = kernel[neg_indices].abs().sum()
                        k = pos_sum / (pos_sum + neg_sum)
                        kernel[pos_indices] *= k
                        kernel[neg_indices] /= 10000

            param.copy_(kernel_params)

        if 'features.5.weight' in name:
            kernel_params = param.clone()
            num_kernels, num_channels, kernel_size, kernel_size = kernel_params.size()

            for i in range(num_kernels):
                for j in range(num_channels):
                    kernel = kernel_params[i, j]
                    pos_indices = kernel > 0
                    neg_indices = kernel < 0
                    if pos_indices.any() or neg_indices.any():
                        pos_sum = kernel[pos_indices].sum()
                        neg_sum = kernel[neg_indices].abs().sum()
                        k = pos_sum / (pos_sum + neg_sum)
                        kernel[pos_indices] *= k
                        kernel[neg_indices] /= 10000

            param.copy_(kernel_params)

        if 'features.6.weight' in name:
            kernel_params = param.clone()
            num_kernels, num_channels, kernel_size, kernel_size = kernel_params.size()

            for i in range(num_kernels):
                for j in range(num_channels):
                    kernel = kernel_params[i, j]
                    pos_indices = kernel > 0
                    neg_indices = kernel < 0
                    if pos_indices.any() or neg_indices.any():
                        pos_sum = kernel[pos_indices].sum()
                        neg_sum = kernel[neg_indices].abs().sum()
                        k = pos_sum / (pos_sum + neg_sum)
                        kernel[pos_indices] *= k
                        kernel[neg_indices] /= 10000

            param.copy_(kernel_params)

        if 'classifier.0.weight'in name:
            #Process the parameters of the fully connected layer
            weight_params = param.clone()
            num_units, num_weights = weight_params.size()

            # Iterate over the units in the fully connected layer.
            for i in range(num_units):
                unit = weight_params[i]
                pos_indices = unit > 0
                neg_indices = unit < 0
                if pos_indices.any() or neg_indices.any():
                    pos_sum = unit[pos_indices].sum()
                    neg_sum = unit[neg_indices].abs().sum()
                    k = pos_sum / (pos_sum + neg_sum)
                    unit[pos_indices] *= k
                    unit[neg_indices] /= 10000

            param.copy_(weight_params)

        if 'classifier.1.weight'in name:
            weight_params = param.clone()
            num_units, num_weights = weight_params.size()

            for i in range(num_units):
                unit = weight_params[i]
                pos_indices = unit > 0
                neg_indices = unit < 0
                if pos_indices.any() or neg_indices.any():
                    pos_sum = unit[pos_indices].sum()
                    neg_sum = unit[neg_indices].abs().sum()
                    k = pos_sum / (pos_sum + neg_sum)
                    unit[pos_indices] *= k
                    unit[neg_indices] /= 10000

            param.copy_(weight_params)

        if 'classifier.2.weight'in name:
            weight_params = param.clone()
            num_units, num_weights = weight_params.size()

            for i in range(num_units):
                unit = weight_params[i]
                pos_indices = unit > 0
                neg_indices = unit < 0
                if pos_indices.any() or neg_indices.any():
                    pos_sum = unit[pos_indices].sum()
                    neg_sum = unit[neg_indices].abs().sum()
                    k = pos_sum / (pos_sum + neg_sum)
                    unit[pos_indices] *= k
                    unit[neg_indices] /= 10000

            param.copy_(weight_params)

torch.save(model.state_dict(), 'modified_model_deReLu_modweight.pth')