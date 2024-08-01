import argparse

import torch
from networks import get_network
from torch import optim
from transformers import TransformerNet
from utils.loading import parse_spec

DEVICE = "cpu"


def loss_TransformerNet(output_clb: torch.Tensor):
    negative_values_index = output_clb < 0
    return max(torch.log(-output_clb[negative_values_index]))


def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    Adam_iterations = 1000
    Adam_lr = 1

    flattened_inputs = inputs.flatten()
    lb = (flattened_inputs - eps).clamp(min = 0, max = 1)
    ub = (flattened_inputs + eps).clamp(min = 0, max = 1)

    transformerNet = TransformerNet(net, true_label, inputs, verbose=False)

    # if the network doesn't contain any relu or leakyrelu layer
    # there is nothing to optimize over
    if not transformerNet.all_linear:
        optimizer = optim.Adam(
            transformerNet.transformers_ModuleList.parameters(), lr=Adam_lr
        )
        optimizer.zero_grad()

    first_iteration = True

    #print("First iteration without Adam")
    output_clb, output_cub = transformerNet(
        lb, ub, first_iteration=first_iteration, print_alphas=False
    )

    if all(output_clb > 0):
        return True

    if transformerNet.all_linear:
        return False

    first_iteration = False

    # if you feel like plotting the losses for some reason
    #losses = []

    for i in range(Adam_iterations):
        loss = loss_TransformerNet(output_clb)
        #losses.append(loss.item())
        #print("Current loss: {}".format(loss.item()))
        loss.backward()
        optimizer.step()
        #print("-" * 80 + "\n")
        #print("next iteration")
        optimizer.zero_grad()
        output_clb, output_cub = transformerNet(
            lb, ub, first_iteration=first_iteration, print_alphas=False
        )
        if all(output_clb > 0):
            return True

    return all(output_clb > 0)


def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument(
        "--spec", type=str, required=True, help="Test case to verify."
    )
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    # print(args.spec)

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(
        DEVICE
    )

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
