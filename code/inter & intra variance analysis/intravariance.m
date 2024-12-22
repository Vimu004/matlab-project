dataFolder = "C:\Users\vithj\OneDrive\Documents\Visual Studio Code\AI ML\CW-Data\CW-Data"; 
files = dir(fullfile(dataFolder, '*.mat')); 

% Sort files based on user ID
fileNames = {files.name}; 
userIDs = regexp(fileNames, 'U(\d+)_', 'tokens'); 
userIDs = cellfun(@(x) str2double(x{1}), userIDs); 
[sortedUserIDs, sortIdx] = sort(userIDs); 
files = files(sortIdx); 

% List of variables to analyze
variablesToAnalyze = {'Acc_FreqD_FDay', 'Acc_TimeD_FDay', 'Acc_FreqD_MDay', 'Acc_TimeD_MDay'};

fprintf('=== STARTING INTRA-VARIANCE ANALYSIS ===\n');

% Initialize a structure to hold user-specific data
userData = struct();


for fileIndex = 1:length(files)
    
    fileName = fullfile(dataFolder, files(fileIndex).name);
    data = load(fileName); 
    
    % Extract user ID and variable name
    [~, fileBaseName, ~] = fileparts(files(fileIndex).name); 
    variableNameParts = split(fileBaseName, '_'); 
    userID = str2double(variableNameParts{1}(2:end)); 
    variableName = strjoin(variableNameParts(2:end), '_');
    
   
    if ismember(variableName, variablesToAnalyze)
        variableData = struct2cell(data); 
        variableData = variableData{:}(:); 
        
        % Calculate intra-user statistics for the variable
        meanValue = mean(variableData); 
        varianceValue = var(variableData); 
        stdDevValue = std(variableData); 
        
        % Store metrics for the user and variable
        if ~isfield(userData, sprintf('User%d', userID))
            userData.(sprintf('User%d', userID)) = struct();
        end
        userData.(sprintf('User%d', userID)).(variableName) = ...
            [meanValue, varianceValue, stdDevValue];
    end
end


userFields = fieldnames(userData); 
for i = 1:length(userFields)
    userName = userFields{i};
    userMetrics = userData.(userName); 
    
    figure('Name', sprintf('Intra-Variance Analysis - %s', userName), 'NumberTitle', 'off');
    variableNames = fieldnames(userMetrics); 
    
    for j = 1:length(variableNames)
        variableName = variableNames{j};
        metrics = userMetrics.(variableName); 
        
        
        subplot(2, 2, j); 
        bar(metrics, 'FaceColor', [0.2 0.6 0.8]); 
        title(variableName, 'Interpreter', 'none');
        xticks(1:3);
        xticklabels({'Mean', 'Variance', 'Std Dev'});
        ylabel('Values');
        xlabel('Metrics');
        grid on;
    end
    
    sgtitle(sprintf('Intra-Variance Analysis for %s', userName)); 
end

fprintf('=== INTRA-VARIANCE ANALYSIS COMPLETE ===\n');
