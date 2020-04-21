credit_default_original = readtable('credit_default_processed.xlsx', 'PreserveVariableNames', true);

X_original = credit_default_original(:,2:29);
y_original = credit_default_original(:,30);

%% Split the dataset randomly into training (80%) and test (20%) set

rng(0);
random_num_original = randperm(length(y_original{:,1}));

X_train_original_tuning = X_original(random_num_original(1:5000),:);
y_train_original_tuning = y_original(random_num_original(1:5000),:);

X_train_original = X_original(random_num_original(1:24000),:);
y_train_original = y_original(random_num_original(1:24000),:);

X_test_original = X_original(random_num_original(24001:30000),:);
y_test_original = y_original(random_num_original(24001:30000),:);

%% Fitting the baseline model to the dataset
% (5000 tuning)

SVMmodel_baseline = fitcsvm(X_train_original, y_train_original);

%% Test the baseline model in unseen data

test_accuracy = sum((predict(SVMmodel_baseline,X_test_original) == y_test_original{:,1}))/length(y_test_original{:,1})*100;
y_predicted = predict(SVMmodel_baseline,X_test_original);
y_label = table2array(y_test_original);

%% Plotting the linear confusion matrix and calculating the f1 score

confusion_matrix = confusionchart(y_label, y_predicted);
confusion_matrix.Title = 'Confusion matrix - baseline model';
confusion_matrix.FontName = 'Cambria';
f1_score = f1_score(y_label, y_predicted);
cohens_kappa = cohens_kappa(y_label, y_predicted);

%% f1 score = 0.3736, cohens kappa = 0.0546, accuracy = 42.9%)
% training 24000

fprintf('The f1 score of the baseline model is: %f\n ', f1_score)
fprintf('The cohens kappa of the network is: %f\n ', cohens_kappa)
fprintf('The accuracy of the network is: %f\n ', test_accuracy)