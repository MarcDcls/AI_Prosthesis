from numpy import genfromtxt

my_data = genfromtxt('corpus_students_only_validated_targets.csv', delimiter=',')
print(my_data[2])
