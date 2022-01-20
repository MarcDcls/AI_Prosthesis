from naive_seq import generate_naive_seq
from data import format_data
from basic_NN import generate_basic_NN

if __name__ == '__main__':
    # generate_basic_NN()

    computed_arm_angles = generate_naive_seq(2000)
    format_data(computed_arm_angles, "res/naive_seq.csv")
