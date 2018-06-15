import process_query
import pprint
from process_query import predict_IOB_labels

s1 = 'Can you please show me flights from new york to los angeles arriving before 6 pm'
s2 = 'I want to see flights from denver to philadelphia departing after 8 pm on monday'

print("\n", s1)
pprint.pprint(predict_IOB_labels(s1))

print("\n", s2)
pprint.pprint(predict_IOB_labels(s2))
