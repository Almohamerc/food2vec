import sys

# Saves the cuisine-ingredient (srep00196-s3.csv) dataset from the Flavor 
# Network and Principles of Food Pairing paper as food2vec.csv, where the 
# first row lists the cuisine (11) and ingredient (381) names and the 
# following (56498) rows are bit vectors corresponding to the 
# cuisine-ingredient rows from the input (srep00196-s3.csv) file.
#
# The script then creates files X.csv and Y.csv: the former contains  
# one-hot vectors u corresponding to each bit vector v of food2vec.csv, and 
# the latter stores the corresponding (context) bit vectors (v - u).
#
# Usage: python prepare_data.py

def main(args):
    fh = open("srep00196-s3.csv", "r")
    lines = fh.readlines()
    fh.close()
    words = set()
    for line in lines[4:]:
        line = line.strip()
        toks = line.split(",")
        for tok in toks:
            words.add(tok)
    sorted_words = sorted(words)
    fh = open("food2vec.csv", "w")
    fh.write("%s\n" %(",".join(sorted_words)))
    for line in lines[4:]:
        line = line.strip()
        toks = line.split(",")
        bits = ["1" if word in toks else "0" for word in sorted_words]
        fh.write("%s\n" %(",".join(bits)))
    fh.close()

    fh = open("food2vec.csv", "r")
    lines = fh.readlines()
    fh.close()
    Xfh = open("X.csv", "w")
    Yfh = open("Y.csv", "w")
    for line in lines[1:]:
        toks = line.strip().split(",")
        hot_indices = [i for i, x in enumerate(toks) if x == "1"]
        for i in hot_indices:
            x_vector = ["0"] * len(toks)
            x_vector[i] = "1"
            y_vector = [tok for tok in toks]
            y_vector[i] = "0"
            Xfh.write("%s\n" %(",".join(x_vector)))
            Yfh.write("%s\n" %(",".join(y_vector)))
    Xfh.close()
    Yfh.close()

if __name__ == "__main__":
    main(sys.argv[1:])
