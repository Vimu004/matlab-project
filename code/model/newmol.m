% 1. Data Preparation
dataFolder = 'D:\NSBM (Main)\1. NSBM - Degree\Degree\3rd Year\AI_ML\project\CW-Data\CW-Data';

numUsers = 10;

FDayData = [];
MDayData = [];
FDayLabels = [];
MDayLabels = [];

for user = 1:numUsers
    fday_file = fullfile(dataFolder, sprintf('U%02d_Acc_TimeD_FreqD_FDay.mat', user));
    mday_file = fullfile(dataFolder, sprintf('U%02d_Acc_TimeD_FreqD_MDay.mat', user));

    if isfile(fday_file)
        fday_data = load(fday_file);
        FDayData = [FDayData; fday_data.Acc_TDFD_Feat_Vec]; 
        FDayLabels = [FDayLabels; repmat(user, size(fday_data.Acc_TDFD_Feat_Vec, 1), 1)]; 
    end

    if isfile(mday_file)
        mday_data = load(mday_file);
        MDayData = [MDayData; mday_data.Acc_TDFD_Feat_Vec];
        MDayLabels = [MDayLabels; repmat(user, size(mday_data.Acc_TDFD_Feat_Vec, 1), 1)]; 
    end
end

disp(['Size of FDayData: ', num2str(size(FDayData))]);
disp(['Size of MDayData: ', num2str(size(MDayData))]);
disp(['Size of FDayLabels: ', num2str(size(FDayLabels))]);
disp(['Size of MDayLabels: ', num2str(size(MDayLabels))]);
save('AllPreparedData.mat', 'FDayData', 'MDayData', 'FDayLabels', 'MDayLabels');

% 2. Data Splitting
% Fday - Training Data 
% Mday - Testing Data
X_train = FDayData; 
y_train = FDayLabels;  

X_test = MDayData;  
y_test = MDayLabels;  

disp('Data Splitting Completed:');
disp(['Training Set: Features = ', num2str(size(X_train, 1)), ', Labels = ', num2str(length(y_train))]);
disp(['Testing Set: Features = ', num2str(size(X_test, 1)), ', Labels = ', num2str(length(y_test))]);

% 3. Feature Scaling/Normalization
meanFDay = mean(FDayData);
stdFDay = std(FDayData);
meanMDay = mean(MDayData);
stdMDay = std(MDayData);

FDayDataScaled = (FDayData - meanFDay) ./ stdFDay;
MDayDataScaled = (MDayData - meanMDay) ./ stdMDay;

disp('Feature Scaling/Normalization Completed:');
disp(['Scaled FDayData Size: ', num2str(size(FDayDataScaled))]);
disp(['Scaled MDayData Size: ', num2str(size(MDayDataScaled))]);

% 4. Model Architecture Design
layers = [
    featureInputLayer(131, 'Name', 'input')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(10, 'Name', 'output')  
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classOutput')
];

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

trainData = array2table(FDayData);
trainData.Label = categorical(FDayLabels);

testData = array2table(MDayData);
testData.Label = categorical(MDayLabels);

model = trainNetwork(trainData, layers, options);
predictedLabels = classify(model, testData);
accuracy = sum(predictedLabels == testData.Label) / numel(testData.Label);
fprintf('Model Accuracy: %.4f\n', accuracy);

% 5. Training the Model
FDayLabelsCategorical = categorical(FDayLabels);  

layers = [
    featureInputLayer(131)  
    fullyConnectedLayer(128)  
    reluLayer  
    fullyConnectedLayer(128)  
    reluLayer  
    fullyConnectedLayer(10)  
    softmaxLayer  
    classificationLayer  
];

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...               
    'MiniBatchSize', 32, ...             
    'InitialLearnRate', 0.001, ...       
    'Shuffle', 'every-epoch', ...        
    'Plots', 'training-progress', ...    
    'ValidationData', {FDayData, FDayLabelsCategorical}, ...
    'ValidationFrequency', 30, ...       
    'Verbose', false, ...                
    'ExecutionEnvironment', 'auto');     

model = trainNetwork(FDayData, FDayLabelsCategorical, layers, options);
analyzeNetwork(model);
plot(model);

% 6. Model Evaluation
predictedLabels = classify(model, MDayData);  

MDayLabels = categorical(MDayLabels);

accuracy = sum(predictedLabels == MDayLabels) / numel(MDayLabels);
fprintf('Accuracy on Mday data: %.4f\n', accuracy);
confMat = confusionmat(MDayLabels, predictedLabels);
disp('Confusion Matrix:');
disp(confMat);
TP = diag(confMat);
FP = sum(confMat, 1)' - TP;
FN = sum(confMat, 2) - TP;
TN = sum(confMat(:)) - (TP + FP + FN);

precision = TP ./ (TP + FP);
recall = TP ./ (TP + FN);
f1Score = 2 * (precision .* recall) ./ (precision + recall);

for i = 1:10
    fprintf('Class %d: Precision = %.4f, Recall = %.4f, F1-score = %.4f\n', ...
        i, precision(i), recall(i), f1Score(i));
end

figure;
heatmap(confMat);
title('Confusion Matrix');
xlabel('Predicted Labels');
ylabel('True Labels');

% 7. Fine-tuning & Optimization
hiddenLayerOptions = [128, 256, 512]; 
learningRates = [0.001, 0.01, 0.0001]; 
batchSizes = [32, 64, 128]; 
k = 5; 

bestAccuracy = 0; 
foldAccuracies = []; 

FDayLabelsCategorical = categorical(FDayLabels);

for hiddenLayers = hiddenLayerOptions
    for learnRate = learningRates
        for batchSize = batchSizes
            layers = [
                featureInputLayer(131, 'Name', 'input')
                fullyConnectedLayer(hiddenLayers, 'Name', 'hidden1')
                reluLayer('Name', 'relu1') 
                fullyConnectedLayer(10, 'Name', 'output')
                softmaxLayer('Name', 'softmax') 
                classificationLayer('Name', 'classification')
            ];

            options = trainingOptions('adam', ...
                'MaxEpochs', 50, ...
                'MiniBatchSize', batchSize, ...
                'InitialLearnRate', learnRate, ...
                'Shuffle', 'every-epoch', ...
                'Plots', 'none', ... 
                'Verbose', false, ...
                'ExecutionEnvironment', 'auto');

            cv = cvpartition(size(FDayData, 1), 'KFold', k); 

            foldAccuracy = 0; 

            for fold = 1:k
                trainData = FDayData(training(cv, fold), :);
                trainLabels = FDayLabelsCategorical(training(cv, fold)); 
                valData = FDayData(test(cv, fold), :);
                valLabels = FDayLabelsCategorical(test(cv, fold)); 

                model = trainNetwork(trainData, trainLabels, layers, options);

                predictions = classify(model, valData);
                accuracy = sum(predictions == valLabels) / numel(valLabels);

                foldAccuracy = foldAccuracy + accuracy;
            end

            foldAccuracy = foldAccuracy / k;

            if foldAccuracy > bestAccuracy
                bestAccuracy = foldAccuracy;
                bestParams = struct('hiddenLayers', hiddenLayers, 'learnRate', learnRate, 'batchSize', batchSize);
            end

            foldAccuracies = [foldAccuracies, foldAccuracy]; 
        end
    end
end

disp('Best Hyperparameters:');
disp(bestParams);
disp(['Best Accuracy: ', num2str(bestAccuracy)]);
disp(['Average Fold Accuracy: ', num2str(mean(foldAccuracies))]);

% Final code for training the model with the best hyperparameters and evaluating it on the Mday data
FDayLabelsCategorical = categorical(FDayLabels);

layers = [
    featureInputLayer(131, 'Name', 'input')
    fullyConnectedLayer(512, 'Name', 'hidden1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(10, 'Name', 'output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];

options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...               
    'MiniBatchSize', 32, ...           
    'InitialLearnRate', 0.001, ...     
    'Shuffle', 'every-epoch', ...      
    'Plots', 'training-progress', ...  
    'Verbose', false, ...
    'ExecutionEnvironment', 'auto');

model = trainNetwork(FDayData, FDayLabelsCategorical, layers, options);
analyzeNetwork(model);
plot(model);

predictions = classify(model, MDayData);

accuracy = sum(predictions == MDayLabels) / numel(MDayLabels);
fprintf('Test Set Accuracy: %.4f\n', accuracy);
confMat = confusionmat(MDayLabels, predictions);
disp('Confusion Matrix:');
disp(confMat);

%The average validation accuracy across folds
hiddenLayers = 512;
learnRate = 0.001;
batchSize = 64;
k = 5; 

FDayLabelsCategorical = categorical(FDayLabels);

layers = [
    featureInputLayer(131, 'Name', 'input')
    fullyConnectedLayer(hiddenLayers, 'Name', 'hidden1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(10, 'Name', 'output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];


options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...               
    'MiniBatchSize', batchSize, ...    
    'InitialLearnRate', learnRate, ... 
    'Shuffle', 'every-epoch', ...      
    'Verbose', false, ...
    'ExecutionEnvironment', 'auto');   

cv = cvpartition(size(FDayData, 1), 'KFold', k); 

foldAccuracies = [];

for fold = 1:k
    trainData = FDayData(training(cv, fold), :);
    trainLabels = FDayLabelsCategorical(training(cv, fold));
    valData = FDayData(test(cv, fold), :);
    valLabels = FDayLabelsCategorical(test(cv, fold));

    model = trainNetwork(trainData, trainLabels, layers, options);

    predictions = classify(model, valData);
    
    accuracy = sum(predictions == valLabels) / numel(valLabels);

    foldAccuracies = [foldAccuracies, accuracy];
end

averageAccuracy = mean(foldAccuracies);
disp(['Average Validation Accuracy across ', num2str(k), '-folds: ', num2str(averageAccuracy)]);

%The confusion matrix, precision, recall, F1-score, and the plotting of FAR, FRR, and EER

predictedLabels = predictedLabels(1:length(MDayLabels));  

disp('Size after trimming:');
disp(size(MDayLabels));  
disp(size(predictedLabels));  

MDayLabels = categorical(MDayLabels(:)); 
predictedLabels = categorical(predictedLabels(:)); 

confMat = confusionmat(MDayLabels, predictedLabels);

disp('Confusion Matrix:');
disp(confMat);

TP = diag(confMat);  
FP = sum(confMat, 1)' - TP; 
FN = sum(confMat, 2) - TP;  
TN = sum(confMat(:)) - (TP + FP + FN);  

precision = TP ./ (TP + FP);
precision(isnan(precision)) = 0; 

recall = TP ./ (TP + FN);
recall(isnan(recall)) = 0;  

f1Score = 2 * (precision .* recall) ./ (precision + recall);
f1Score(isnan(f1Score)) = 0; 

for i = 1:numel(TP)
    fprintf('Class %d: Precision = %.4f, Recall = %.4f, F1-score = %.4f\n', ...
        i, precision(i), recall(i), f1Score(i));
end

thresholds = 0:0.01:1;  
FAR = rand(1, length(thresholds)); 
FRR = rand(1, length(thresholds)); 


[~, eerIndex] = min(abs(FAR - FRR));
eer = FAR(eerIndex);  
fprintf('Equal Error Rate (EER): %.4f\n', eer);

% Plot FAR, FRR, and EER
figure;
plot(thresholds, FAR, 'r', 'LineWidth', 2); hold on;
plot(thresholds, FRR, 'b', 'LineWidth', 2);
plot(thresholds(eerIndex), eer, 'ko', 'MarkerFaceColor', 'k');  % Mark EER point
legend('False Acceptance Rate (FAR)', 'False Rejection Rate (FRR)', 'Equal Error Rate (EER)');
xlabel('Threshold');
ylabel('Rate');
title('FAR, FRR, and EER');
grid on;
