# TODOs
# in decreasing order of priority

# 1) Implement the Conv2dTransformer --> done, kind of
# 2) Implement slope optimization (requires completing LeakyReluTransformer too) -->done
# 3) Turn matrix multiplication needed for backsub for leaky relu layer into
#   componentwise multiplication and summation to speedup forward --> would be cool to do
# 4) Remove other minor efficiencies from the code


import time
from typing import Optional

# Temporary, might not be necessary?
import numpy as np
import torch
import torch.nn.functional as F
from conv_utils import calculate_toeplitz
from torch import nn


def construct_backsub_matrices(weight_lb, bias_lb, weight_ub, bias_ub):
    """
    builds the matrices needed for backsub for relu
    and leakyrelu transformers
    """
    weight_cat_bias_lb = torch.cat(
        [weight_lb, torch.unsqueeze(bias_lb, dim=1)], dim=1
    )

    weight_cat_bias_ub = torch.cat(
        [weight_ub, torch.unsqueeze(bias_ub, dim=1)], dim=1
    )

    row = torch.zeros(weight_cat_bias_lb.shape[1])
    row[-1] = 1
    row = torch.unsqueeze(row, dim=0)

    weight_cat_bias_backsub_lb = torch.cat([weight_cat_bias_lb, row], dim=0)
    weight_cat_bias_backsub_ub = torch.cat([weight_cat_bias_ub, row], dim=0)
    return (
        weight_cat_bias_backsub_lb,
        weight_cat_bias_backsub_ub,
        weight_cat_bias_lb,
        weight_cat_bias_ub,
    )


def construct_backsub_matrix(weight, bias):
    """
    builds the matrices need during backsub for linear
    and conv2d transformers
    """
    weight_cat_bias_lb = torch.cat(
        [weight, torch.unsqueeze(bias, dim=1)], dim=1
    )
    weight_cat_bias_ub = weight_cat_bias_lb

    row = torch.zeros(weight_cat_bias_lb.shape[1])
    row[-1] = 1
    row = torch.unsqueeze(row, dim=0)

    weight_cat_bias_backsub_lb = torch.cat([weight_cat_bias_lb, row], dim=0)
    weight_cat_bias_backsub_ub = weight_cat_bias_backsub_lb
    return (
        weight_cat_bias_backsub_lb,
        weight_cat_bias_backsub_ub,
        weight_cat_bias_lb,
        weight_cat_bias_ub,
    )


def backsub(lb, ub, weight_cat_bias_backsub_lb, weight_cat_bias_backsub_ub):
    """
    the operations to backsobstitute to the previus layer
    """

    lb_positive = F.relu(lb)

    lb_negative = F.relu(-lb)

    ub_positive = F.relu(ub)

    ub_negative = F.relu(-ub)

    new_lb = torch.matmul(
        lb_positive, weight_cat_bias_backsub_lb
    ) - torch.matmul(lb_negative, weight_cat_bias_backsub_ub)

    new_ub = torch.matmul(
        ub_positive, weight_cat_bias_backsub_ub
    ) - torch.matmul(ub_negative, weight_cat_bias_backsub_lb)

    return new_lb, new_ub


class InputTransformer(nn.Module):
    """
    All transformers are now subclass of nn.Module.
    Besides the old methods, they now have forward
    that is used to perform backsubstitution
    and get_initial_backsub_matrices, which
    I just defined just for conveninece and
    clarity


    This tranformer in particular represents the input
    concrete constrains and cotains the matrices to perform
    the last backsub step
    """

    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        super(InputTransformer, self).__init__()
        self.weight_cat_bias_lb = lb.clone().detach_()
        self.weight_cat_bias_ub = ub.clone().detach_()

        self.weight_cat_bias_backsub_lb = torch.cat(
            [self.weight_cat_bias_lb, torch.ones(1)], dim=0
        )
        self.weight_cat_bias_backsub_ub = torch.cat(
            [self.weight_cat_bias_ub, torch.ones(1)], dim=0
        )

    def create_backsub_matrices(
        self, lb: torch.Tensor, ub: torch.Tensor, first_iteration=True
    ):
        pass

    def forward(self, lb, ub):
        """
        All forward methods of transformes use the backsub function, they perform
        the same operation but each with its own matrices
        """
        new_lb, new_ub = backsub(
            lb,
            ub,
            self.weight_cat_bias_backsub_lb,
            self.weight_cat_bias_backsub_ub,
        )
        return new_lb, new_ub

    def get_initial_backsub_matrices(self):
        return self.weight_cat_bias_lb, self.weight_cat_bias_ub


class LinearTransformer(nn.Module):
    def __init__(self, layer: nn.Linear):
        super(LinearTransformer, self).__init__()
        self.weight = layer.weight.clone().detach_()
        self.bias = layer.bias.clone().detach_()

        (
            self.weight_cat_bias_backsub_lb,
            self.weight_cat_bias_backsub_ub,
            self.weight_cat_bias_lb,
            self.weight_cat_bias_ub,
        ) = construct_backsub_matrix(self.weight, self.bias)

    def create_backsub_matrices(
        self, lb: torch.Tensor, ub: torch.Tensor, first_iteration=True
    ):
        pass

    def forward(self, lb, ub):
        new_lb, new_ub = backsub(
            lb,
            ub,
            self.weight_cat_bias_backsub_lb,
            self.weight_cat_bias_backsub_ub,
        )
        return new_lb, new_ub

    def get_initial_backsub_matrices(self):
        return self.weight_cat_bias_lb, self.weight_cat_bias_ub


class Conv2dTransformer(nn.Module):
    def __init__(
        self, layer: nn.Conv2d, input_size: tuple, output_size: tuple
    ):
        super(Conv2dTransformer, self).__init__()
        kernel = layer.weight.clone().detach_()
        bias = layer.bias.clone().detach_()
        self.weight = calculate_toeplitz(
            kernel,
            input_size,
            # Double check: Is padding and stride always symmetric?
            stride=layer.stride[0],
            padding=layer.padding[0],
        )
        _, out_width, out_height = output_size
        self.bias = torch.repeat_interleave(
            bias, out_width * out_height
        ).detach()
        (
            self.weight_cat_bias_backsub_lb,
            self.weight_cat_bias_backsub_ub,
            self.weight_cat_bias_lb,
            self.weight_cat_bias_ub,
        ) = construct_backsub_matrix(self.weight, self.bias)

    def forward(self, lb, ub):
        new_lb, new_ub = backsub(
            lb,
            ub,
            self.weight_cat_bias_backsub_lb,
            self.weight_cat_bias_backsub_ub,
        )
        return new_lb, new_ub

    def create_backsub_matrices(
        self, lb: torch.Tensor, ub: torch.Tensor, first_iteration=True
    ):
        pass

    def get_initial_backsub_matrices(self):
        return self.weight_cat_bias_lb, self.weight_cat_bias_ub


class ReluTransformer(nn.Module):
    """
    The (hopefully) definitive version of the (non leaky)
    relu layer. One of its attributes is alphas, a tensor
    that contains the slopes to be used for the lower bound when
    there is a crossing happening. This tensor will be passed
    to Adam to perform slope optimization. At each gradient step,
    the values of alphas will be updated.

    Before applying Adam to optimize the slopes, this transformer
    applies the smallest area heuristic to compute an initial value
    for the slopes.
    """

    def __init__(self, layer: nn.ReLU, input_features: int):
        super(ReluTransformer, self).__init__()
        self.weight_cat_bias_lb = None
        self.weight_cat_bias_ub = None
        self.weight_cat_bias_backsub_lb = None
        self.weight_cat_bias_backsub_ub = None
        self.alphas = nn.Parameter(torch.zeros(input_features))

    def compute_lambda_and_intercept(lb: float, ub: float):
        """
        this is a utility method that computes the slope (lambda) and the
        intercept of the line that constitute the upper constraint when the relu
        is crossing
        """
        lambda_ = ub / (ub - lb)
        return lambda_, -(lambda_ * lb)

    def create_backsub_matrices(
        self, clb: torch.Tensor, cub: torch.Tensor, first_iteration=True
    ):
        # safety clamp, since the gradient step might produce
        # invalid slopes
        with torch.no_grad():
            self.alphas.clamp_(min=0, max=1)

        diag_entries_lb = torch.zeros_like(clb)
        diag_entries_ub = torch.zeros_like(cub)
        bias_ub = torch.zeros_like(cub)
        bias_lb = torch.zeros_like(clb)

        for i in range(len(clb)):
            if cub[i] <= 0:
                diag_entries_lb[i] = 0
                diag_entries_ub[i] = 0
            elif clb[i] >= 0:
                diag_entries_lb[i] = 1
                diag_entries_ub[i] = 1
            else:
                if first_iteration:
                    if abs(clb[i]) > abs(cub[i]):
                        with torch.no_grad():
                            self.alphas[i] = 0
                        diag_entries_lb[i] = self.alphas[i]
                    else:
                        with torch.no_grad():
                            self.alphas[i] = 1
                        diag_entries_lb[i] = self.alphas[i]
                else:
                    diag_entries_lb[i] = self.alphas[i]
                (
                    lambda_,
                    intercept,
                ) = ReluTransformer.compute_lambda_and_intercept(
                    clb[i], cub[i]
                )
                diag_entries_ub[i] = lambda_
                bias_ub[i] = intercept

        self.weight_lb = torch.diag(diag_entries_lb)
        self.weight_ub = torch.diag(diag_entries_ub)

        (
            self.weight_cat_bias_backsub_lb,
            self.weight_cat_bias_backsub_ub,
            self.weight_cat_bias_lb,
            self.weight_cat_bias_ub,
        ) = construct_backsub_matrices(
            self.weight_lb, bias_lb, self.weight_ub, bias_ub
        )

    def forward(self, lb, ub):
        new_lb, new_ub = backsub(
            lb,
            ub,
            self.weight_cat_bias_backsub_lb,
            self.weight_cat_bias_backsub_ub,
        )
        return new_lb, new_ub

    def get_initial_backsub_matrices(self):
        return self.weight_cat_bias_lb, self.weight_cat_bias_ub


class LeakyReluTransformer(nn.Module):
    """
    similarly to the ReluTransformer, before applying Adam to optimize the
    slopes this transformer applies the smallest area heuristic to compute
    the slopes for crossing components.
    """

    def __init__(self, layer: nn.LeakyReLU, input_features: int):
        super(LeakyReluTransformer, self).__init__()
        self.negative_slope = layer.negative_slope
        self.weight_cat_bias_lb = None
        self.weight_cat_bias_ub = None
        self.weight_cat_bias_backsub_lb = None
        self.weight_cat_bias_backsub_ub = None
        self.alphas = nn.Parameter(
            torch.zeros(input_features) + self.negative_slope
        )

    def compute_lambda_and_intercept(self, lb: float, ub: float):
        """
        this is a utility method that computes the slope (lambda) and the
        intercept of the line that constitute the upper constraint when the leaky
        relu is crossing
        """
        lambda_ = (ub - (lb * self.negative_slope)) / (ub - lb)
        intercept = ((self.negative_slope - 1) * ub * lb) / (ub - lb)
        return lambda_, intercept

    def create_backsub_matrices(
        self, clb: torch.Tensor, cub: torch.Tensor, first_iteration=True
    ):
        # safety clamp
        # again, it's not a given that the gradient step wil produce valid slopes
        if self.negative_slope <= 1:
            with torch.no_grad():
                self.alphas.clamp_(min=self.negative_slope, max=1)

        else:
            with torch.no_grad():
                self.alphas.clamp_(min=1, max=self.negative_slope)

        diag_entries_lb = torch.zeros_like(clb)
        diag_entries_ub = torch.zeros_like(cub)
        bias_ub = torch.zeros_like(cub)
        bias_lb = torch.zeros_like(clb)

        for i in range(len(clb)):
            if cub[i] <= 0:
                diag_entries_lb[i] = self.negative_slope
                diag_entries_ub[i] = self.negative_slope
            elif clb[i] >= 0:
                diag_entries_lb[i] = 1
                diag_entries_ub[i] = 1
            else:
                lambda_, intercept = self.compute_lambda_and_intercept(
                    clb[i], cub[i]
                )
                if self.negative_slope <= 1:
                    diag_entries_ub[i] = lambda_
                    bias_ub[i] = intercept
                    if first_iteration:
                        if abs(clb[i]) > abs(cub[i]):
                            with torch.no_grad():
                                self.alphas[i] = self.negative_slope
                            diag_entries_lb[i] = self.alphas[i]
                        else:
                            with torch.no_grad():
                                self.alphas[i] = 1
                            diag_entries_lb[i] = self.alphas[i]
                    else:
                        diag_entries_lb[i] = self.alphas[i]

                else:
                    diag_entries_lb[i] = lambda_
                    bias_lb[i] = intercept
                    if first_iteration:
                        if abs(clb[i]) > abs(cub[i]):
                            with torch.no_grad():
                                self.alphas[i] = self.negative_slope
                            diag_entries_ub[i] = self.alphas[i]
                        else:
                            with torch.no_grad():
                                self.alphas[i] = 1
                            diag_entries_ub[i] = self.alphas[i]
                    else:
                        diag_entries_ub[i] = self.alphas[i]

        self.weight_lb = torch.diag(diag_entries_lb)
        self.weight_ub = torch.diag(diag_entries_ub)

        (
            self.weight_cat_bias_backsub_lb,
            self.weight_cat_bias_backsub_ub,
            self.weight_cat_bias_lb,
            self.weight_cat_bias_ub,
        ) = construct_backsub_matrices(
            self.weight_lb, bias_lb, self.weight_ub, bias_ub
        )

    def forward(self, lb, ub):
        new_lb, new_ub = backsub(
            lb,
            ub,
            self.weight_cat_bias_backsub_lb,
            self.weight_cat_bias_backsub_ub,
        )
        return new_lb, new_ub

    def get_initial_backsub_matrices(self):
        return self.weight_cat_bias_lb, self.weight_cat_bias_ub


class VerificationTransformer(nn.Module):
    """
    The final transformer of the verifying network. It computes the 9
    differences of the logits corresponding to the true label of the sample
    and all other 9 logits
    """

    def __init__(self, true_label: int):
        super(VerificationTransformer, self).__init__()
        self.weight = torch.zeros(9, 10)
        self.weight[:, true_label] = 1
        row = 0
        for col in range(10):
            if col == true_label:
                continue
            self.weight[row, col] = -1
            row += 1
        self.bias = torch.zeros(9)

        (
            self.weight_cat_bias_backsub_lb,
            self.weight_cat_bias_backsub_ub,
            self.weight_cat_bias_lb,
            self.weight_cat_bias_ub,
        ) = construct_backsub_matrix(self.weight, self.bias)

    def create_backsub_matrices(
        self, clb: torch.Tensor, cub: torch.Tensor, first_iteration=True
    ):
        pass

    def forward(self, lb, ub):
        raise Exception("This should not be called")

    def get_initial_backsub_matrices(self):
        return self.weight_cat_bias_lb, self.weight_cat_bias_ub


class TransformerNet(nn.Module):
    """
    This object underwent quite a significant makeover, but
    at the end of the day it performs the same operations as before

    !!! Verbose now just prints time for each operation and the concrete
    bounds for each layer, since
    printing all the matrices is useless, too big too gain any insight
    by visualization. time useful to compare different solution, concrete bounds
    useful for sanity checks

    """

    def __init__(
        self,
        net: nn.Module,
        true_label: int,
        in_feature: np.ndarray,
        verbose: Optional[bool] = False,
    ):
        super(TransformerNet, self).__init__()
        starting_time = time.time()
        self.verbose = verbose
        # self.transformers_list = []
        self.clb_list = []
        self.cub_list = []
        # self.backsub_matrices_lb = []
        # self.backsub_matrices_ub = []
        self.transformers_ModuleList = nn.ModuleList([])
        self.all_linear = True

        if self.verbose:
            print("-" * 40 + "__init__" + "-" * 40 + "\n")

        network_layers_list = [
            layer
            for layer in net.modules()
            if type(layer) is not nn.Sequential
        ]

        for i, layer in enumerate(network_layers_list):
            in_feature_shape = in_feature.shape
            gt_layer_output = layer(in_feature)

            # don't know if we need this commented if
            # if type(layer) == networks.Normalization:
            # pass
            if type(layer) is nn.Flatten:
                """
                if verbose:
                    print("There is a Flatten layer")
                    print()
                    print("-" * 88 + "\n")
                """
                in_feature = in_feature.flatten()
                continue
            elif type(layer) is nn.Linear:
                # self.transformers_list.append(LinearTransformer(layer))
                self.transformers_ModuleList.append(LinearTransformer(layer))
            elif type(layer) is nn.ReLU:
                # self.transformers_list.append(ReluTransformer(layer, len(in_feature.view(-1)) ))
                self.transformers_ModuleList.append(
                    ReluTransformer(layer, len(in_feature.view(-1)))
                )
                self.all_linear = False
            elif type(layer) is nn.LeakyReLU:
                # self.transformers_list.append(LeakyReluTransformer(layer, len(in_feature.view(-1))))
                self.transformers_ModuleList.append(
                    LeakyReluTransformer(layer, len(in_feature.view(-1)))
                )
                self.all_linear = False
            elif type(layer) is nn.Conv2d:
                # self.transformers_list.append(
                #    Conv2dTransformer(
                #        layer,
                #        input_size=in_feature_shape,
                #        output_size=gt_layer_output.shape,
                #    )
                # )

                self.transformers_ModuleList.append(
                    Conv2dTransformer(
                        layer,
                        input_size=in_feature_shape,
                        output_size=gt_layer_output.shape,
                    )
                )
            else:
                raise Exception(
                    "Layer "
                    + str(type(layer))
                    + "has not a corresponding transformer yet"
                )

            # Set input for the next layer
            in_feature = gt_layer_output
            """
            if self.verbose:
                print(
                    "Transformer n. {}: ".format(i + 1)
                    + str(type(self.transformers_list[-1]))
                )
                print()
                print("weight_cat_bias_lb: ")
                print(self.transformers_list[-1].weight_cat_bias_lb)
                print()
                print("weight_cat_bias_ub: ")
                print(self.transformers_list[-1].weight_cat_bias_ub)
                print()
                print("weight_cat_bias_backsub_lb: ")
                print(self.transformers_list[-1].weight_cat_bias_backsub_lb)
                print()
                print("weight_cat_bias_backsub_ub: ")
                print(self.transformers_list[-1].weight_cat_bias_backsub_ub)
                print()
                print("-" * 88 + "\n")
            """

        # self.transformers_list.append(VerificationTransformer(true_label))
        self.transformers_ModuleList.append(
            VerificationTransformer(true_label)
        )

        # self.transformers_ModuleList = nn.ModuleList(self.transformers_list)
        """
        if self.verbose:
            print(
                "Verification Transformer: "
                + str(type(self.transformers_list[-1]))
            )
            print()
            print("weight_cat_bias_lb: ")
            print(self.transformers_list[-1].weight_cat_bias_lb)
            print()
            print("weight_cat_bias_ub: ")
            print(self.transformers_list[-1].weight_cat_bias_ub)
            print()
            print("weight_cat_bias_backsub_lb: ")
            print(self.transformers_list[-1].weight_cat_bias_backsub_lb)
            print()
            print("weight_cat_bias_backsub_ub: ")
            print(self.transformers_list[-1].weight_cat_bias_backsub_ub)
            print()
            print("-" * 36 + "__init__ completed" + "-" * 36 + "\n")
            print("Total time: {} \n".format(time.time()-starting_time))
            print("-" * 40 + "--------" + "-" * 40 + "\n\n\n")
        """
        if self.verbose:
            print("Total time: {} \n".format(time.time() - starting_time))
            print("-" * 40 + "--------" + "-" * 40 + "\n\n\n")

    def forward(
        self,
        clb: torch.Tensor,
        cub: torch.Tensor,
        first_iteration=True,
        print_alphas=True,
    ):
        """
        this implementation assumes that we always want to backsobstitute to the
        input "layer", and peforms the full chain of matrix multiplications anew
        for each layer.

        Takes as input the lower bound tensor and the upper bound tensor.
        Returns final lower bound tensor and final upper bound tensor
        """
        starting_time = time.time()

        if self.verbose:
            print("-" * 40 + "forward" + "-" * 40 + "\n")
            print("Input lb:")
            print(clb)
            print()
            print("Input ub:")
            print(cub)
            print()
            print("-" * 88 + "\n")

        # these list contain the concrete ub and lb for each layer
        # the first entry is the initial tensor of lb and ub passed as arguments
        # maybe we only need the concrete constrains of last layer, but
        # keeping everything might be useful for debugging

        if first_iteration:
            # self.transformers_list.insert(0, InputTransformer(clb, cub))
            self.transformers_ModuleList.insert(0, InputTransformer(clb, cub))

        self.clb_list = []
        self.cub_list = []
        self.clb_list.append(clb)
        self.cub_list.append(cub)

        # these contain the matrices necessary to perform backsub
        # backsub is performed by multiplying the matrices bacwards without
        # transposing them

        # I think this will become obsolete
        """
      self.backsub_matrices_lb = []
      self.backsub_matrices_ub = []
      self.backsub_matrices_lb.append(torch.cat([clb, torch.ones(1)], dim=0))
      self.backsub_matrices_ub.append(torch.cat([cub, torch.ones(1)], dim=0))
      """

        """
      if self.verbose:
          print("Layer 0 (before first layer of the network)\n")
          print("clb_list:")
          print(self.clb_list)
          print()
          print("cub_list:")
          print(self.cub_list)
          print()
          print("backsub_matrices_lb:")
          print(self.backsub_matrices_lb[-1])
          print()
          print("backsub_matrices_ub:")
          print(self.backsub_matrices_ub[-1])
          print()
          print("-" * 88 + "\n")
       """

        for n, transformer in enumerate(self.transformers_ModuleList):
            # not the most elegant thing
            if n == 0:
                continue

            # fetch the backsub matrices of the current transformer, the one without
            # the extra 00...001 row
            transformer.create_backsub_matrices(
                self.clb_list[-1], self.cub_list[-1], first_iteration
            )

            """
          weight_cat_bias_lb = transformer.weight_cat_bias_lb
          weight_cat_bias_ub = transformer.weight_cat_bias_ub

          new_clb = weight_cat_bias_lb

          new_cub = weight_cat_bias_ub
          """
            new_clb, new_cub = transformer.get_initial_backsub_matrices()

            # do the mat multiplicattion to the very beginning
            for i in range(n - 1, -1, -1):
                current_transformer = self.transformers_ModuleList[i]
                new_clb, new_cub = current_transformer(new_clb, new_cub)

            # add the newly computed concrete bounds to the list
            self.clb_list.append(new_clb)
            self.cub_list.append(new_cub)

            # add the backsub matrices to the list, this time the version enhanced with
            # the extra 000...0001 row
            # this is not needed for the last transformer, for now let's keep it

            # became obsolete
            """
          weight_cat_bias_backsub_lb = transformer.weight_cat_bias_backsub_lb
          weight_cat_bias_backsub_ub = transformer.weight_cat_bias_backsub_ub

          self.backsub_matrices_lb.append(weight_cat_bias_backsub_lb)
          self.backsub_matrices_ub.append(weight_cat_bias_backsub_ub)
          """

            if self.verbose:
                print("Layer {}\n".format(n + 1))
                print("clb_list:")
                print(self.clb_list)
                print()
                print("cub_list:")
                print(self.cub_list)
                print()
                """
              print("backsub_matrices_lb:")
              for mat in self.backsub_matrices_lb:
                  print(mat)
                  print()
              print("\n")
              print("backsub_matrices_ub:")
              for mat in self.backsub_matrices_ub:
                  print(mat)
                  print()
              print()
              """

                print("-" * 88 + "\n")

        if self.verbose:
            print("-" * 35 + "forward completed" + "-" * 35 + "\n")
            print("Final concrete lb:")
            print(self.clb_list[-1])
            print()
            print("Total time: {} sec".format(time.time() - starting_time))
            print()
            print("-" * 88 + "\n")
        if print_alphas:
            print("Parameters (alphas) :")
            for p in self.transformers_ModuleList.parameters():
                print(p)
            print("-" * 88 + "\n")

        return self.clb_list[-1], self.cub_list[-1]
