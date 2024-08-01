import os
import subprocess
import time

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


def run_verifier():
    net_list = [
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
    ]

    # Load a txt file
    gt_file = open("test_cases/gt.txt", "r")
    # Get the content of the file
    gt = gt_file.read()
    gt = gt.split("\n")

    verify_results = []

    for net in net_list:
        spec_folder = os.path.join("test_cases", net)
        # List the files in the folder
        spec_list = os.listdir(spec_folder)
        spec_paths = [os.path.join(spec_folder, spec) for spec in spec_list]
        for spec in spec_paths:
            print(f"Verifying {net} with {spec}...")
            command = [
                "python",
                "code/verifier.py",
                "--net",
                net,
                "--spec",
                spec,
            ]
            start_time = time.time()
            result = subprocess.run(command, stdout=subprocess.PIPE)
            # result = subprocess.run(command)
            end_time = time.time()
            print(f"Time taken: {end_time - start_time}")
            verify_or_not = result.stdout.decode("utf-8").split("\n")[-2]
            verify_results.append(verify_or_not)

            # Get ground truth.
            for gt_lines in gt:
                gt_net, gt_spec, gt_result = gt_lines.split(",")
                if gt_net != net:
                    continue
                if gt_spec not in spec:
                    continue
                print(
                    f"    Code runs: {verify_or_not}. Took"
                    f" {end_time - start_time} seconds."
                )
                if verify_or_not == gt_result:
                    print(GREEN + "    Correct!" + RESET)
                else:
                    print(RED + "    Wrong!" + RESET)


if __name__ == "__main__":
    run_verifier()
