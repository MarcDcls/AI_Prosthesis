from naive_seq import generate_naive_seq, generate_interpolated_seq
from smart_seq import generate_smart_seq, generate_smart_generative_seq
from data import format_data
from basic_NN import generate_basic_NN
from predictive_NN import generate_predictive_NN

if __name__ == '__main__':
    # generate_basic_NN()

    # naive_seq, hands = generate_naive_seq(2000)
    # format_data(naive_seq, "res/naive_seq.csv")
    # format_data(naive_seq, "res/naive_seq_with_hands.csv", add_cols=hands)

    # interpolated_seq = generate_interpolated_seq(2000)
    # format_data(interpolated_seq, "res/interpolated_seq.csv")

    # generate_predictive_NN()

    # smart_seq = generate_smart_seq(2000)
    # format_data(smart_seq, "res/smart_seq.csv")
    #
    # smart_generative_seq = generate_smart_generative_seq(2000)
    # format_data(smart_generative_seq, "res/smart_generative_seq.csv")