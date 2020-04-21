function f1 = f1_score(y_real, y_predicted)
    true_positive = sum((y_predicted == 1) & (y_real == 1)); 
    false_positive = sum((y_predicted == 1) & (y_real == 0));
    false_negative = sum((y_predicted == 0) & (y_real == 1));
    precision = true_positive / (true_positive + false_positive);
    recall = true_positive  / (true_positive + false_negative);
    f1 = (2 * precision * recall) / (precision + recall);
end