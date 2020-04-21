% https://uk.mathworks.com/matlabcentral/fileexchange/69943-simple-cohen-s-kappa

function kappa = cohens_kappa(y_real, y_predicted)
    confusion_mat = confusionmat(y_real, y_predicted); % compute confusion matrix
    n = sum(confusion_mat(:)); % get total N
    confusion_mat = confusion_mat./n; % Convert confusion matrix counts to proportion of n
    rows_agg = sum(confusion_mat,2); % row sum
    columns_agg = sum(confusion_mat); % column sum
    expected = rows_agg *columns_agg; % expected proportion for random agree
    proportion_correct = sum(diag(confusion_mat)); % Observed proportion correct
    proportion_expected = sum(diag(expected)); % Proportion correct expected
    kappa = (proportion_correct-proportion_expected)/(1-proportion_expected); % Cohen's kappa
end