credit_default_adasyn = readtable('credit_default_ADASYN(15_best_features_original).xlsx', 'PreserveVariableNames', true);
credit_default_original = readtable('credit_default_15_best_features_original.xlsx', 'PreserveVariableNames', true);

X_original = credit_default_original(:,1:15);
y_original = credit_default_original(:,16);

X_adasyn = credit_default_adasyn(:,2:16);
y_adasyn = credit_default_adasyn(:,17);

%% Split the dataset randomly into training (80%) and test (20%) set

rng(0);
random_num_original = randperm(length(y_original{:,1}));

rng(2);
random_num_adasyn = randperm(length(y_adasyn{:,1}));

X_train_adasyn = X_adasyn(random_num_adasyn(1:24000),:);
X_train_adasyn = transpose(X_train_adasyn{:,:});
y_train_adasyn = y_adasyn(random_num_adasyn(1:24000),:);
y_train_adasyn = transpose(y_train_adasyn{:,:});

X_test_original = X_original(random_num_original(24001:30000),:);
X_test_original = transpose(X_test_original{:,:});
y_test_original = y_original(random_num_original(24001:30000),:);
y_test_original = transpose(y_test_original{:,:});

%% Third MLP architecture (15 node input layer, 3 hidden layers (13 nodes), 1 node output layer)
% Training and hyperparameter tuning of the model

learning_rate = [0.1, 0.01, 0.001, 0.0001];
momentum = [0.5, 0.7, 0.9, 0.95, 0.99];
activation_function_hidden = {'elliotsig', 'logsig', 'tansig'};
activation_function_output = {'elliotsig', 'logsig', 'tansig', 'softmax'};
net3_best_perf = 100;


tic
timerVal = tic;
    for l = 1:length(learning_rate)
        for m = 1:length(momentum)
            for afh = 1:length(activation_function_hidden)
                for afo = 1:length(activation_function_output)
                    net3 = patternnet([13 13 13]);
                    net3.trainFcn = 'traingdx';
                    net3.outputs{4}.processFcns = {};
                    net3.layers{1}.transferFcn = activation_function_hidden{afh};
                    net3.layers{2}.transferFcn = activation_function_output{afh};
                    net3.layers{3}.transferFcn = activation_function_output{afh};
                    net3.layers{4}.transferFcn = activation_function_output{afo};
                    net3.performFcn = 'crossentropy';
                    net3.divideParam.trainRatio = 70/100;
                    net3.divideParam.valRatio   = 30/100;
                    net3.divideParam.testRatio  = 0/100;
                    net3.trainParam.epochs = 100;
                    net3.trainParam.lr = learning_rate(l);
                    net3.trainParam.mc = momentum(m);
                    net3.trainParam.showCommandLine = true;
                    net3.trainParam.time = inf; 
                    rng(3);
                    net3_trained = train(net3, X_train_adasyn, y_train_adasyn);
                    y_predicted = net3_trained(X_train_adasyn);
                    perf = crossentropy(net3_trained, y_train_adasyn, y_predicted);
                    if perf < net3_best_perf
                        net3_best_perf = perf;
                        net3_best_lr = learning_rate(l);
                        net3_best_mom = momentum(m);
                        net3_best_af_h = activation_function_hidden(afh);
                        net3_best_af_o = activation_function_output(afo);
                    end
                end
            end
        end
    end

fprintf('The optimal activation function of the hidden layers is: %s\n ', net3_best_af_h{1});
fprintf('The optimal activation function of the output layer is: %s\n ', net3_best_af_o{1});
fprintf('The optimal learning rate is: %f\n', net3_best_lr);
fprintf('The optimal momentum is: %f\n', net3_best_mom);

toc
elapsedTime = toc(timerVal);
net3_time = elapsedTime/60;

%% Training again the best model given by crossentropy loss function (best values for its hyperparameters) 
% activation function from input to hidden layer: "tansig"
% activation function from hidden to output layer: "logsig"
% learning rate: 0.1
% momentum: 0.7

best_net3 = patternnet([13 13 13], 'traingdx');
best_net3.outputs{4}.processFcns = {};
best_net3.layers{1}.transferFcn = net3_best_af_h{1};
best_net3.layers{2}.transferFcn = net3_best_af_h{1};
best_net3.layers{3}.transferFcn = net3_best_af_h{1};
best_net3.layers{4}.transferFcn = net3_best_af_o{1};
best_net3.performFcn = 'crossentropy';
best_net3.divideParam.trainRatio = 70/100;
best_net3.divideParam.valRatio   = 30/100;
best_net3.divideParam.testRatio  = 0/100;
best_net3.trainParam.epochs = 300;
best_net3.trainParam.lr = net3_best_lr;
best_net3.trainParam.mc = net3_best_mom;
rng(3);
best_net3_trained = train(best_net3, X_train_adasyn, y_train_adasyn);

%% Testing the best net3 model given by crossentropy loss function for unseen data

y_label = transpose(y_test_original);
best_net3_y_predicted = transpose(round(best_net3_trained(X_test_original)));
net3_confusion_matrix = confusionchart(y_label, best_net3_y_predicted);
net3_confusion_matrix.Title = 'Confusion Matrix - net3 MLP - Test Data - adasyn training';
net3_confusion_matrix.FontName = 'Cambria';
f1_net3 = f1_score(y_label, best_net3_y_predicted);
cohens_kappa_net3 = cohens_kappa(y_label, best_net3_y_predicted);
test_accuracy_net3 = sum(best_net3_y_predicted == y_label)/length(y_label)*100;

%% (f1 score = 0.5078, cohens kappa = 0.3461, accuracy = 75.38%, training time = 7.1)

fprintf('The f1 score of the network is: %f\n ', f1_net3)
fprintf('The cohens kappa of the network is: %f\n ', cohens_kappa_net3)
fprintf('The accuracy of the network is: %f\n ', test_accuracy_net3)
fprintf('The time elapsed in minutes for the training of the network is: %f\n ', net3_time)
