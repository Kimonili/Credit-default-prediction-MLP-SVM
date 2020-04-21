credit_default_original = readtable('credit_default_processed.xlsx', 'PreserveVariableNames', true);

% split the dataset to predictors and target variables
X_original = credit_default_original(:,2:29); 
y_original = credit_default_original(:,30);

%% Split the dataset randomly into training (80%) and test (20%) set

rng(0); % for reproducibility
random_num_original = randperm(length(y_original{:,1})); % randomised numbers from 1 to the length of the dataset

% creating the training set which is 80% of the dataset
X_train_original = X_original(random_num_original(1:24000),:); % randomised predictors matrix 
X_train_original = transpose(X_train_original{:,:}); % transpose the predictor matrix for later use
y_train_original = y_original(random_num_original(1:24000),:); % randomised target array
y_train_original = transpose(y_train_original{:,:}); % transposed the target array for later use

% creating the test set which is the rest 20% of the dataset
X_test_original = X_original(random_num_original(24001:30000),:); % randomised predictors matrix 
X_test_original = transpose(X_test_original{:,:}); % transpose the predictor matrix for later use
y_test_original = y_original(random_num_original(24001:30000),:); % randomised target array 
y_test_original = transpose(y_test_original{:,:}); % transposed the target array for later use
%% Baseline model MLP training
% Training and hyperparameter tuning of the model

tic
timerVal = tic;
net_baseline = patternnet(); % creating our feedforward neural network
rng(3); % for reproducibility (initializing the same weights every time)
net_baseline_trained = train(net_baseline, X_train_original, y_train_original); % train the model
toc
elapsedTime = toc(timerVal);
net_baseline_time = elapsedTime/60; % calculating the time passed for training in minutes

%% Testing the best net1 model given by crossentropy loss function for unseen data

y_label = transpose(y_test_original);
net_baseline_y_predicted = transpose(round(net_baseline_trained(X_test_original))); % fitting the test data to the trained model and assign it a variable
net_baseline_confusion_matrix = confusionchart(y_label, net_baseline_y_predicted); % confusiom matrix
net_baseline_confusion_matrix.Title = 'Confusion Matrix - baseline network - Test Data';
net_baseline_confusion_matrix.FontName = 'Cambria';
f1_net_baseline = f1_score(y_label, net_baseline_y_predicted); % f1 score
cohens_kappa_net_baseline = cohens_kappa(y_label, net_baseline_y_predicted); % cohens kappa
test_accuracy_net_baseline = sum(net_baseline_y_predicted == y_label)/length(y_label)*100;% accuracy

%% (f1 score = 0.4044, cohens kapa = 0.3038, accuracy = 80.31%, time = 0.03) % results

fprintf('The f1 score of the network is: %f\n ', f1_net_baseline)
fprintf('The cohens kappa of the network is: %f\n ', cohens_kappa_net_baseline)
fprintf('The accuracy of the network is: %f\n ', test_accuracy_net_baseline)
fprintf('The time elapsed in minutes for the training of the network is: %f\n ', net_baseline_time)