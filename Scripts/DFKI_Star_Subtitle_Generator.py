import csv
with open("Video11.srt", "w") as f_out, open("./TVL11.csv", "r") as f_labels:
    csv_reader = csv.reader(f_labels, delimiter=',')
    for r_i, row in enumerate(csv_reader):
        print(row[2])
        if r_i == 0:
            continue
        f_out.write("\n" + str(r_i-1) + "\n")
        f_out.write(row[0] + " --> " + row[1] + "\n")
        f_out.write(row[2] + "\n")
