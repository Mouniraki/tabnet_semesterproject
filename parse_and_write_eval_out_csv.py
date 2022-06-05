import re
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="", type=str)
    parser.add_argument("--output_file", default="", type=str)
    return parser.parse_args()

def main():
    # Get the command-line arguments
    args = get_args()

    # Opens the output file
    oup = open(args.output_file, mode='w')

    # Write the first row of the file
    oup.write("model[n_steps|n_shared|n_ind|eps_val|n_attack_iters],accuracy,cost-restricted,average-attack-cost,success_rate,rob_accuracy\n")

    # Filters everything that isn't a number/a decimal
    regex = re.compile('[^0-9.|]')

    # Opens the input file, performs a filtering and writes the filtered content to a CSV-file
    with open(args.input_file, 'r') as inp:
        # Skip the first two header lines
        next(inp)
        next(inp)
        for line in inp:
            if "../" in line:
                continue
            elif "Rob Acc" in line:
                oup.write(regex.sub('', line))
            elif "---" in line:
                oup.write('\n')
            else:
                oup.write(regex.sub('', line)+',')

        inp.close()
    oup.close()

if __name__ == "__main__":
    main()