% import the original dataset without feature selection
credit_default_original = readtable('credit_default_processed.xlsx', 'PreserveVariableNames', true);

% import the ADASYN dataset without feature selection
credit_default_adasyn = readtable('credit_default_ADASYN(all_features).xlsx', 'PreserveVariableNames', true);

X_original = credit_default_original(:,2:29); % predictors for testing
y_original = credit_default_original(:,30); % target for testing 

X_adasyn = credit_default_adasyn(:,2:29); % predictors for training
y_adasyn = credit_default_adasyn(:,30); % target for testing

%% Split the dataset randomly into training (80%) and test (20%) set
% The ADASYN balanced dataset is only used for training
% The original imbalanced dataset is only used for testing

rng(0); % reprodicibility
random_num_original = randperm(length(y_original{:,1})); % randomising variable for testing dataset
rng(2); % reproducibility
random_num_adasyn = randperm(length(y_adasyn{:,1})); % randomising variable for training dataset

X_test_original = X_original(random_num_original(24001:30000),:); % predictors for testing
y_test_original = y_original(random_num_original(24001:30000),:); % target for testing 

% We will use 5000 rows out of 24000 that is 80% of the dataset for
% hyperparameter tuning because of high computational cost
X_train_adasyn_tuning = X_adasyn(random_num_adasyn(1:5000),:); % subset of predictors for tuning 
y_train_adasyn_tuning = y_adasyn(random_num_adasyn(1:5000),:); % subset of target for tuning

% We will use 24000 rows (all the training set) to train the best model
% after the tuning
X_train_adasyn = X_adasyn(random_num_adasyn(1:24000),:); % predictors for training
y_train_adasyn = y_adasyn(random_num_adasyn(1:24000),:); % target for training

%% CV partition of the training set

cvp = cvpartition(5000, 'KFold', 10); % 10 fold cross validation

%% Optimization details for the linear kernel

outlierfraction = 0.05; % outliers
% options of the fitcsvm function
% Bayesian optimization as an optimization method
opts =  struct('Optimizer', 'bayesopt', 'ShowPlots', true, 'CVPartition', cvp,...
                    'AcquisitionFunctionName', 'expected-improvement-plus');
                
%% Best Hyperparameters for linear kernel

SVMmodel = fitcsvm(X_train_adasyn_tuning, y_train_adasyn_tuning, 'KernelFunction', 'linear',...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', opts, 'Standardize',true,...
    'OutlierFraction', outlierfraction); % fitting the model to variable and hyperparameter tuning applied

%% Extracting the optimal hypermarameter values from the tuning 

best_boxconstraint = SVMmodel.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint; % box constraint
best_kernelscale = SVMmodel.HyperparameterOptimizationResults.XAtMinObjective.KernelScale; % kernel scale

fprintf('The optimal box contraint value is: %f\n ', best_boxconstraint);
fprintf('The optimal kernel scale values is: %f\n ', best_kernelscale);
%% Fitting the model with the best hyperparameters and training it again in the whole training set
% In case of limited time, I imported the best values of each hyper parameter
% as numbers for the box contraint and the kernel scale inside the fitcsvm
% function. The hyperparameter tuning in the previous code block, should give you the same results.
% best_boxconstraint = 0.001026, best_kernelscale = 1.0857 (24000 training) 

best_SVM_model = fitcsvm(X_train_adasyn, y_train_adasyn, 'KernelFunction', 'linear',...
    'Standardize', true, 'OutlierFraction', 0.05,...
    'BoxConstraint', 0.001026,...
    'KernelScale', 1.0857);

%% Test the best linear kernel model in unseen data

test_accuracy = sum((predict(best_SVM_model, X_test_original) == y_test_original{:,1}))/length(y_test_original{:,1})*100; % accuracy
y_predicted = predict(best_SVM_model, X_test_original); % assign model's predictions to an array
y_label = table2array(y_test_original);

%% Plotting the linear confusion matrix and calculating the f1 score and cohens kappa

confusion_matrix = confusionchart(y_label, y_predicted); % confusion matrix
confusion_matrix.Title = 'Confusion matrix - linear kernel - original test data - adasyn training';
confusion_matrix.FontName = 'Cambria';
f1_score = f1_score(y_label, y_predicted); % f1 score
cohens_kappa = cohens_kappa(y_label, y_predicted); % cohens kappa

%% Plot ROC curve and calculating AUC

[X,Y,T,AUC] = perfcurve(y_label, y_predicted, 1); % ROC and AUC
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by best Support Vector Machine')

%% f1_score = 0.5297, cohens kappa = 0.4002, accuracy = 79.6%, training time = 62.5)
% tuning 5000 rows
% training 24000 rows
% BEST MODEL OF ALL SVMs, WITH TRAINING IN ADASYN DATASET WITH ALL
% THE FEATURES BEING TRAINED (no feature selection)

fprintf('The f1 score of the best linear model is: %f\n ', f1_score)
fprintf('The cohens kappa of the best linear model is: %f\n ', cohens_kappa)
fprintf('The accuracy of the best linear model is: %f\n ', test_accuracy)