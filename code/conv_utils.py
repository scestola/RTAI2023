import time

import numpy as np
import scipy.linalg as linalg
import torch
import torch.nn as nn
import torch.nn.functional as F


def toeplitz_conv1(kernel, input_size):
    k_h, k_w = kernel.shape
    i_h, i_w = input_size
    o_h, o_w = i_h - k_h + 1, i_w - k_w + 1

    # For each 1d convolution, construct a toeplitz matrix.
    toeplitz = [
        linalg.toeplitz(
            c=(kernel[r, 0], *np.zeros(i_w - k_w)),
            r=(*kernel[r], *np.zeros(i_w - k_w)),
        )
        for r in range(k_h)
    ]

    block_toeplitz = get_block_toeplitz(o_h, i_h, toeplitz)

    return block_toeplitz


def get_block_toeplitz(num_h_blocks, num_w_blocks, toeplitz):
    block_height, block_width = toeplitz[0].shape

    # Initializing the convolution matrix
    block_toeplitz = np.zeros(
        (num_h_blocks, block_height, num_w_blocks, block_width)
    )

    # Populating the convolution matrix
    for idx, block in enumerate(toeplitz):
        for height_index in range(num_h_blocks):
            block_toeplitz[height_index, :, idx + height_index, :] = block

    conv_out_h, conv_out_w = (
        num_h_blocks * block_height,
        num_w_blocks * block_width,
    )
    block_toeplitz = block_toeplitz.reshape(conv_out_h, conv_out_w)
    return block_toeplitz


def init_toeplitz(kernel, input_size, padding=1):
    i_c, h, w = input_size
    k_n, k_c, k_h, k_w = kernel.shape

    output_size = (
        k_n,
        h - (k_h - 1) + 2 * padding,
        w - (k_w - 1) + 2 * padding,
    )

    out_n, out_h, out_w = output_size
    toeplitz = np.zeros(
        (
            out_n,
            int(out_h * out_w),
            i_c,
            int(h * w),
        )
    )
    return toeplitz, output_size


def get_toeplitz_with_pad(
    input_size, output_size, toeplitz_matrix, kernel, padding
):
    _, i_h, i_w = input_size

    # Create a padded matrix
    initial_x_offset = padding * (i_w + 2 * padding) + padding
    initial_y_offset = 0
    padded_matrix = np.zeros(
        ((i_h + 2 * padding) * (i_w + 2 * padding), i_h * i_w)
    )

    # Iterating to populate the padded matrix
    x_offset = initial_x_offset
    y_offset = initial_y_offset
    for row in range(i_h):
        padded_matrix[
            x_offset : x_offset + i_w, y_offset : y_offset + i_w
        ] = np.identity(i_w)
        x_offset += i_w + 2 * padding
        y_offset += i_w

    # Now calculate the padded toeplitz
    k_n, k_c, k_h, k_w = kernel.shape
    padded_h, padded_w = i_h + 2 * padding, i_w + 2 * padding

    for i, ks in enumerate(kernel):
        for j, k in enumerate(ks):
            toeplitz_1 = toeplitz_conv1(k, (padded_h, padded_w))
            toeplitz_matrix[i, :, j, :] = toeplitz_1 @ padded_matrix

    return toeplitz_matrix.reshape(np.prod(output_size), np.prod(input_size))


def handle_stride(input_size, toeplitz_matrix, kernel, stride, padding):
    """Returns a matrix that selects the correct output elements
    when multiplied with the output of a toeplitz matrix"""
    _, h, w = input_size
    k_n, k_c, k_h, k_w = kernel.shape

    # Expected output size
    h_out = h - (k_h - 1) + 2 * padding
    w_out = w - (k_w - 1) + 2 * padding

    col_mask = np.zeros(w_out)
    col_mask[::stride] = 1

    # Create a 2D mask array
    mask_2d = np.zeros((h_out, w_out), dtype=np.float32)
    mask_2d[::stride, :] = col_mask
    flatten_mask = mask_2d.flatten()
    cat_mask = np.concatenate([flatten_mask] * k_n, axis=0)
    return toeplitz_matrix[cat_mask > 0]


def calculate_toeplitz(kernel, input_size, stride, padding=1):
    toeplitz, output_size = init_toeplitz(kernel, input_size, padding=padding)
    toeplitz = get_toeplitz_with_pad(
        input_size, output_size, toeplitz, kernel, padding
    )
    toeplitz = handle_stride(input_size, toeplitz, kernel, stride, padding)
    return torch.from_numpy(toeplitz.astype("float32"))


class Conv(nn.Module):
    def __init__(self, input_size, conv_layers, fc_layers, n_class=10):
        super(Conv, self).__init__()

        layers = []
        prev_channels = 3

        for n_channels, kernel_size, stride, padding in conv_layers:
            layers += [
                nn.Conv2d(
                    prev_channels,
                    n_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def reconstruct_conv(in_c, in_h, in_w, k_oc, k_ic, k_h, k_w, padding, stride):
    k = np.random.randn(k_oc * k_ic * k_h * k_w).reshape(
        (k_oc, k_ic, k_h, k_w)
    )
    in_tensor = np.random.randn(in_c, in_h, in_w)

    ta_conv = Conv(in_h, [(k_oc, k_ic, stride, padding)], 0)
    ta_conv.layers[0].weight = nn.Parameter(torch.tensor(k).float())
    ta_conv.layers[0].bias = nn.Parameter(
        torch.tensor(np.zeros(ta_conv.layers[0].bias.shape)).float()
    )
    bias = ta_conv.layers[0].bias
    ta_out = ta_conv(torch.tensor(in_tensor).float()).detach().numpy()
    # print("ground truth shape", ta_out.shape)
    out_channel, out_height, out_width = ta_out.shape

    T = (
        calculate_toeplitz(k, in_tensor.shape, stride, padding=padding)
        .detach()
        .numpy()
    )

    mat_bias = (
        torch.repeat_interleave(bias, out_width * out_height).detach().numpy()
    )
    out = T.dot(in_tensor.flatten())
    bias_out = out + mat_bias
    reshape_out = bias_out.reshape((1, out_channel, out_height, out_width))

    return np.max(np.abs(ta_out - reshape_out))


def test_conv():
    # in_c, in_h, in_w = 1, 9, 9
    # k_oc, k_ic, k_h, k_w = 2, 1, 3, 3  # kernel out channel has some bugs
    # padding = 2
    # stride = 2

    for i in range(20):
        in_c = np.random.randint(1, 3)
        # in_c = 16
        in_h = np.random.randint(10, 32)
        # in_h = 32
        in_w = in_h
        k_oc = np.random.randint(1, 64)
        # k_oc = 64
        k_ic = in_c
        k_h = np.random.randint(1, 10)
        k_w = k_h
        padding = np.random.randint(1, 3)
        stride = np.random.randint(1, 3)
        # Calculate the time taken by calculating error
        time_start = time.time()
        error = reconstruct_conv(
            in_c, in_h, in_w, k_oc, k_ic, k_h, k_w, padding, stride
        )
        time_end = time.time()
        print(f"Error at iteration {i}: {error}")
        print(f"    Time taken: {time_end - time_start}")


if __name__ == "__main__":
    test_conv()
