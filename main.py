from naive_seq import generate_naive_seq, generate_interpolated_seq
from data import format_data
from basic_NN import generate_basic_NN

if __name__ == '__main__':
    # generate_basic_NN()

    interpolated_arm_angles = generate_interpolated_seq(2000)
    format_data(interpolated_arm_angles, "res/interpolated_seq.csv")

    # computed_arm_angles, hands = generate_naive_seq(2000)
    # format_data(computed_arm_angles, "res/naive_seq.csv")
    # format_data(computed_arm_angles, "res/naive_seq_with_hands.csv", add_cols=hands)
