credit_default_original = readtable('credit_default_15_best_features_original.xlsx', 'PreserveVariableNames', true);

X_original = credit_default_original(:,1:15);
y_original = credit_default_original(:,16);

%% Split the dataset randomly into training (80%) and test (20%) set

rng(0);
random_num_original = randperm(length(y_original{:,1}));

X_train_original = X_original(random_num_original(1:24000),:);
X_train_original = transpose(X_train_original{:,:});
y_train_original = y_original(random_num_original(1:24000),:);
y_train_original = transpose(y_train_original{:,:});

X_test_original = X_original(random_num_original(24001:30000),:);
X_test_original = transpose(X_test_original{:,:});
y_test_original = y_original(random_num_original(24001:30000),:);
y_test_original = transpose(y_test_original{:,:});

%% Second MLP architecture (15 node input layer, 2 hidden layers (13 nodes), 1 node output layer)
% Training and hyperparameter tuning of the model

learning_rate = [0.1, 0.01, 0.001, 0.0001];
momentum = [0.5, 0.7, 0.9, 0.95, 0.99];
activation_function_hidden = {'elliotsig', 'logsig', 'tansig'};
activation_function_output = {'elliotsig', 'logsig', 'tansig', 'softmax'};
net2_best_perf = 100;


tic
timerVal = tic;
    for l = 1:length(learning_rate)
        for m = 1:length(momentum)
            for afh = 1:length(activation_function_hidden)
                for afo = 1:length(activation_function_output)
                    net2 = patternnet([13 13]);
                    net2.trainFcn = 'traingdx';
                    net2.outputs{3}.processFcns = {};
                    net2.layers{1}.transferFcn = activation_function_hidden{afh};
                    net2.layers{2}.transferFcn = activation_function_output{afh};
                    net2.layers{3}.transferFcn = activation_function_output{afo};
                    net2.performFcn = 'crossentropy';%performance_function{performance};
                    net2.divideParam.trainRatio = 70/100; 
                    net2.divideParam.valRatio = 30/100;
                    net2.divideParam.testRatio = 0/100;
                    net2.trainParam.epochs = 100;
                    net2.trainParam.lr = learning_rate(l);
                    net2.trainParam.mc = momentum(m);
                    net2.trainParam.showCommandLine = true;
                    net2.trainParam.time = inf; 
                    rng(3);
                    net2_trained = train(net2, X_train_original, y_train_original);
                    y_predicted = net2_trained(X_train_original);
                    perf = crossentropy(net2_trained, y_train_original, y_predicted);
                    if perf < net2_best_perf
                        net2_best_perf = perf;
                        net2_best_lr = learning_rate(l);
                        net2_best_mom = momentum(m);
                        net2_best_af_h = activation_function_hidden(afh);
                        net2_best_af_o = activation_function_output(afo);
                    end
                end
            end
        end
    end

fprintf('The optimal activation function of the hidden layers is: %s\n ', net2_best_af_h{1});
fprintf('The optimal activation function of the output layer is: %s\n ', net2_best_af_o{1});
fprintf('The optimal learning rate is: %f\n', net2_best_lr);
fprintf('The optimal momentum is: %f\n', net2_best_mom);

toc
elapsedTime = toc(timerVal);
net2_time = elapsedTime/60;

%% Training again the best model given by crossentropy loss function (best values for its hyperparameters) 
% activation function from input to hidden layer: "tansig"
% activation function from hidden to output layer: "logsig"
% learning rate: 0.1
% momentum: 0.9

best_net2 = patternnet([13 13], 'traingdx');
best_net2.outputs{3}.processFcns = {};
best_net2.layers{1}.transferFcn = net2_best_af_h{1};
best_net2.layers{2}.transferFcn = net2_best_af_h{1};
best_net2.layers{3}.transferFcn = net2_best_af_o{1};
best_net2.performFcn = 'crossentropy';
best_net2.divideParam.trainRatio = 70/100; 
best_net2.divideParam.valRatio = 30/100;
best_net2.divideParam.testRatio = 0/100;
best_net2.trainParam.epochs = 300;
best_net2.trainParam.lr = net2_best_lr;
best_net2.trainParam.mc = net2_best_mom;
rng(3);
best_net2_trained = train(best_net2, X_train_original, y_train_original);

%% Testing the best net2 model given by crossentropy loss function for unseen data

y_label = transpose(y_test_original);
best_net2_y_predicted = transpose(round(best_net2_trained(X_test_original)));
net2_confusion_matrix = confusionchart(y_label, best_net2_y_predicted);
net2_confusion_matrix.Title = 'Confusion Matrix - net2 MLP - Test Data - training original';
net2_crossentropy_confusion_matrix.FontName = 'Cambria';
f1_net2 = f1_score(y_label, best_net2_y_predicted);
cohens_kappa_net2 = cohens_kappa(y_label, best_net2_y_predicted);
test_accuracy_net2 = sum(best_net2_y_predicted == y_label)/length(y_label)*100;

%% (f1 score = 0.4174, cohens kapa = 0.3165, accuracy = 80.5%, time = 4.3)

fprintf('The f1 score of the network is: %f\n ', f1_net2)
fprintf('The cohens kappa of the network is: %f\n ', cohens_kappa_net2)
fprintf('The accuracy of the network is: %f\n ', test_accuracy_net2)
fprintf('The time elapsed in minutes for the training of the network is: %f\n ', net2_time)
