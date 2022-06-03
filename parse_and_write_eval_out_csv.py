import re

# Opens the output file
oup = open("ieeecis_adv_eval_out2.csv", mode='w')

# Write the first row of the file
oup.write("model[n_steps|n_shared|n_ind|eps_val|n_attack_iters],accuracy,cost-restricted,average-attack-cost,success_rate,rob_accuracy\n")

# Filters everything that isn't a number/a decimal
regex = re.compile('[^0-9.|]')

# Opens the input file, performs a filtering and writes the filtered content to a CSV-file
with open("ieeecis_adv_eval_2.txt", 'r') as inp:
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
