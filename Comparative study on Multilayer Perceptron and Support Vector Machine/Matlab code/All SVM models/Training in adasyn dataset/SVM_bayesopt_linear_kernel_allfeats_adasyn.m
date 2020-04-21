credit_default_original = readtable('credit_default_processed.xlsx', 'PreserveVariableNames', true);
credit_default_adasyn = readtable('credit_default_ADASYN(all_features).xlsx', 'PreserveVariableNames', true);

X_original = credit_default_original(:,2:29);
y_original = credit_default_original(:,30);

X_adasyn = credit_default_adasyn(:,2:29);
y_adasyn = credit_default_adasyn(:,30);

%% Split the dataset randomly into training (80%) and test (20%) set

rng(0);
random_num_original = randperm(length(y_original{:,1}));
rng(2);
random_num_adasyn = randperm(length(y_adasyn{:,1}));

X_test_original = X_original(random_num_original(24001:30000),:);
y_test_original = y_original(random_num_original(24001:30000),:);

X_train_adasyn_tuning = X_adasyn(random_num_adasyn(1:5000),:);
y_train_adasyn_tuning = y_adasyn(random_num_adasyn(1:5000),:);

X_train_adasyn = X_adasyn(random_num_adasyn(1:24000),:);
y_train_adasyn = y_adasyn(random_num_adasyn(1:24000),:);

%% CV partition of the training set

cvp = cvpartition(5000, 'KFold', 10);

%% Optimization details for the linear kernel

outlierfraction = 0.05;
opts =  struct('Optimizer', 'bayesopt', 'ShowPlots', true, 'CVPartition', cvp,...
                    'AcquisitionFunctionName', 'expected-improvement-plus');
                
%% Best Hyperparameters for linear kernel

SVMmodel = fitcsvm(X_train_adasyn_tuning, y_train_adasyn_tuning, 'KernelFunction', 'linear',...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', opts, 'Standardize',true,...
    'OutlierFraction', outlierfraction);
%% Extracting the optimal hypermarameter values from the tuning 

best_boxconstraint = SVMmodel.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint;
best_kernelscale = SVMmodel.HyperparameterOptimizationResults.XAtMinObjective.KernelScale;

fprintf('The optimal box contraint value is: %f\n ', best_boxconstraint);
fprintf('The optimal kernel scale values is: %f\n ', best_kernelscale);
%% Fitting the model with the best hyperparameters and training it again
% box_constraint = 0.001026, kernel_scale = 1.0857 (24000 training)

best_SVM_model = fitcsvm(X_train_adasyn, y_train_adasyn, 'KernelFunction', 'linear',...
    'Standardize', true, 'OutlierFraction', 0.05,...
    'BoxConstraint', best_boxconstraint,...
    'KernelScale', best_kernelscale);

%% Test the linear kernel model in unseen data

test_accuracy = sum((predict(best_SVM_model, X_test_original) == y_test_original{:,1}))/length(y_test_original{:,1})*100;
y_predicted = predict(best_SVM_model, X_test_original);
y_label = table2array(y_test_original);

%% Plotting the linear confusion matrix and calculating the f1 score

confusion_matrix = confusionchart(y_label, y_predicted);
confusion_matrix.Title = 'Confusion matrix - linear kernel - original test data - adasyn training';
confusion_matrix.FontName = 'Cambria';
f1_score = f1_score(y_label, y_predicted);
cohens_kappa = cohens_kappa(y_label, y_predicted);
%% Plot ROC curve and calculating AUC

[X,Y,T,AUC] = perfcurve(y_label, y_predicted, 1);
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by best Support Vector Machine')

%% f1_score = 0.5297, cohens kappa = 0.4002, accuracy = 79.6%, training time = 62.5)
% tuning 5000
% training 24000
% BEST MODEL OF ALL SVMs, WITH TRAINING IN ADASYN DATASET WITH ALL
% THE FEATURES DATA BEING TRAINED

fprintf('The f1 score of the best linear model is: %f\n ', f1_score)
fprintf('The cohens kappa of the best linear model is: %f\n ', cohens_kappa)
fprintf('The accuracy of the best linear model is: %f\n ', test_accuracy)