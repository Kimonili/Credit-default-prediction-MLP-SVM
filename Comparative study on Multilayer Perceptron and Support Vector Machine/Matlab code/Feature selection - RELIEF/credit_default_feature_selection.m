%% Import the xlsx file to a table

df = readtable('credit_default_processed.xlsx', 'PreserveVariableNames', true);
df = df(:, 2:30);
%% Create two arrays for the application of relief algorithm. One for the predictors  and one for the target variable

relief_X = table2array(df(:, 1:28)); % predictors's array
relief_Y = table2array(df(:, 29)); % target array

%% Relief algorithm for feature selection 
% https://uk.mathworks.com/help/stats/relieff.html#mw_e1d60cb2-d78e-4dc0-865f-b9b5d47e2d50

[idx, weights] = relieff(relief_X, relief_Y, 10); % applying the relief algorithm

%% Bar plot presenting the best features 


bar(weights(idx)); % bar plot with the importance of each feature descendingly sorted
xlabel('Predictor rank');
ylabel('Predictor importance weight');
%% Keeping the best features 

best_indexes = idx(1:15); % keeping the 15 most important features 
bestX_Y = df(:,[best_indexes, 29]);
writetable(bestX_Y,'credit_default_15_best_features_original.xlsx'); % export them as an excel file
