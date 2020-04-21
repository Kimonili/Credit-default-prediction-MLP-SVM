% import the dataset without feature selection
credit_default_original = readtable('credit_default_processed.xlsx', 'PreserveVariableNames', true);

X_original = credit_default_original(:,2:29); % predictors
y_original = credit_default_original(:,30); % target

%% Split the dataset randomly into training (80%) and test (20%) set

rng(0); % for reproducibility
random_num_original = randperm(length(y_original{:,1}));

X_train_original = X_original(random_num_original(1:24000),:); % 80% predictors for training
X_train_original = transpose(X_train_original{:,:});
y_train_original = y_original(random_num_original(1:24000),:);% 80% target for training
y_train_original = transpose(y_train_original{:,:});

X_test_original = X_original(random_num_original(24001:30000),:); % 20% predictors for testing
X_test_original = transpose(X_test_original{:,:});
y_test_original = y_original(random_num_original(24001:30000),:); % 20% target for testing
y_test_original = transpose(y_test_original{:,:});
%% First MLP architecture (25 node input layer, 1 hidden layers (25 nodes), 1 node output layer)
% Training and hyperparameter tuning of the model

% hyperparameter values tuned
learning_rate = [0.1, 0.01, 0.001, 0.0001]; % learning eate
momentum = [0.5, 0.7, 0.9, 0.95, 0.99]; % momentum
activation_function_hidden = {'elliotsig', 'logsig', 'tansig'}; % activation functions for the hidden layers
activation_function_output = {'elliotsig', 'logsig', 'tansig', 'softmax'}; % activation functions for the output layer
net1_best_perf = 100; % initializing the network's performance (cross entropy)


tic
timerVal = tic;
    for l = 1:length(learning_rate) % 
        for m = 1:length(momentum)
            for afh = 1:length(activation_function_hidden)
                for afo = 1:length(activation_function_output)
                    net1 = patternnet(25); % create feedforward network with 1 hidden and 25 nodes
                    net1.trainFcn = 'traingdx'; % training function
                    net1.outputs{2}.processFcns = {}; % no process function for the output layer
                    net1.layers{1}.transferFcn = activation_function_hidden{afh}; % activation function for hidden layer
                    net1.layers{2}.transferFcn = activation_function_output{afo}; % activation function for output layer
                    net1.performFcn = 'crossentropy'; % performance metric cross entropy
                    net1.divideParam.trainRatio = 70/100; % 70% of the training set used for training
                    net1.divideParam.valRatio = 30/100; % 30% of the training set used for validation
                    net1.divideParam.testRatio = 0/100; % no test set
                    net1.trainParam.epochs = 100; % number of epochs
                    net1.trainParam.lr = learning_rate(l);
                    net1.trainParam.mc = momentum(m);
                    net1.trainParam.showCommandLine = true;
                    net1.trainParam.time = inf; 
                    rng(3); % initializing the weights for reproducibility
                    net1_trained = train(net1, X_train_original, y_train_original); % train the net
                    y_predicted = net1_trained(X_train_original); % net's results
                    perf = crossentropy(net1_trained, y_train_original, y_predicted); % cross entropy score
                    if perf < net1_best_perf % if statement for keeping every time the network with the best cross entropy score
                        net1_best_perf = perf;
                        net1_best_lr = learning_rate(l);
                        net1_best_mom = momentum(m);
                        net1_best_af_h = activation_function_hidden(afh);
                        net1_best_af_o = activation_function_output(afo);
                    end
                end
            end
        end
    end
 
fprintf('The optimal activation function of the hidden layer is: %s\n ', net1_best_af_h{1});
fprintf('The optimal activation function of the output layer is: %s\n ', net1_best_af_o{1});
fprintf('The optimal learning rate is: %f\n', net1_best_lr);
fprintf('The optimal momentum is: %f\n', net1_best_mom);

toc
elapsedTime = toc(timerVal);
net1_time = elapsedTime/60;

%% Training again the best model given by crossentropy loss function (best values for its hyperparameters) 
% best activation function from input to hidden layer: "elliotsig"
% best activation function from hidden to output layer: "logsig"
% best learning rate: 0.1
% best momentum: 0.9

best_net1 = patternnet(25, 'traingdx');
best_net1.outputs{2}.processFcns = {};
best_net1.layers{1}.transferFcn = net1_best_af_h{1};
best_net1.layers{2}.transferFcn = net1_best_af_o{1};
best_net1.performFcn = 'crossentropy';
best_net1.divideParam.trainRatio = 70/100; 
best_net1.divideParam.valRatio = 30/100;
best_net1.divideParam.testRatio = 0/100;
best_net1.trainParam.epochs = 10000;
best_net1.trainParam.lr = net1_best_lr;
best_net1.trainParam.mc = net1_best_mom;
rng(3);
best_net1_trained = train(best_net1, X_train_original, y_train_original);

%% Testing the best net1 model given by crossentropy loss function for unseen data

y_label = transpose(y_test_original);
best_net1_y_predicted = transpose(round(best_net1_trained(X_test_original)));
net1_confusion_matrix = confusionchart(y_label, best_net1_y_predicted);
net1_confusion_matrix.Title = 'Confusion Matrix - net1 MLP - Test Data - training original';
net1_confusion_matrix.FontName = 'Cambria';
f1_net1 = f1_score(y_label, best_net1_y_predicted);
cohens_kappa_net1 = cohens_kappa(y_label, best_net1_y_predicted);
test_accuracy_net1 = sum(best_net1_y_predicted == y_label)/length(y_label)*100;

%% (f1 score = 0.4234, cohens kapa = 0.3219, accuracy = 80.6%, time = 5.5) 

fprintf('The f1 score of the network is: %f\n ', f1_net1)
fprintf('The cohens kappa of the network is: %f\n ', cohens_kappa_net1)
fprintf('The accuracy of the network is: %f\n ', test_accuracy_net1)
fprintf('The time elapsed in minutes for the training of the network is: %f\n ', net1_time)
