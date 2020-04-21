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

%% CV partition of the training set

cvp = cvpartition(5000, 'KFold', 10);

%% Optimization details for the linear kernel

outlierfraction = 0.05;
opts =  struct('Optimizer', 'bayesopt', 'ShowPlots', true, 'CVPartition', cvp,...
                    'AcquisitionFunctionName', 'expected-improvement-plus');
                
%% Best Hyperparameters for linear kernel
% (5000 tuning)

SVMmodel_linear = fitcsvm(X_train_original_tuning, y_train_original_tuning, 'KernelFunction', 'linear',...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', opts, 'Standardize',true,...
    'OutlierFraction', outlierfraction);

%% Extracting the optimal hypermarameter values from the tuning 
% best_boxconstraint = 0.038719, best_kernelscale = 2.9941 

best_boxconstraint = SVMmodel_linear.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint;
best_kernelscale = SVMmodel_linear.HyperparameterOptimizationResults.XAtMinObjective.KernelScale;

fprintf('The optimal box contraint value is: %f\n ', best_boxconstraint);
fprintf('The optimal kernel scale values is: %f\n ', best_kernelscale);

%% Fitting the model with the best hyperparameters and training it again
% best_boxconstraint = 0.038719, best_kernelscale = 2.9941 

best_SVM_model = fitcsvm(X_train_original, y_train_original, 'KernelFunction', 'linear',...
    'Standardize', true, 'OutlierFraction', 0.05,...
    'BoxConstraint', best_boxconstraint,...
    'KernelScale', best_kernelscale);

%% Test the linear kernel model in unseen data

test_accuracy = sum((predict(best_SVM_model,X_test_original) == y_test_original{:,1}))/length(y_test_original{:,1})*100;
y_predicted = predict(best_SVM_model,X_test_original);
y_label = table2array(y_test_original);

%% Plotting the linear confusion matrix and calculating the f1 score

confusion_matrix = confusionchart(y_label, y_predicted);
confusion_matrix.Title = 'Confusion matrix - linear kernel - original test data - training original';
confusion_matrix.FontName = 'Cambria';
f1_score = f1_score(y_label, y_predicted);
cohens_kappa = cohens_kappa(y_label, y_predicted);

%% f1 score = 0.4987, cohens kappa = 0.3854, accuracy = 80.9%, training time = 85.9)
% tuning 5000
% training 24000

fprintf('The f1 score of the best linear model is: %f\n ', f1_score)
fprintf('The cohens kappa of the best linear model is: %f\n ', cohens_kappa)
fprintf('The accuracy of the best linear model is: %f\n ', test_accuracy)