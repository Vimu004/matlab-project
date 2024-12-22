dataFolder = 'C:\Users\vithj\OneDrive\Documents\Visual Studio Code\AI ML\CW-Data\CW-Data'; 
files = dir(fullfile(dataFolder, '*.mat')); 

fileNames = {files.name}; 
userIDs = regexp(fileNames, 'U(\d+)_', 'tokens'); 
userIDs = cellfun(@(x) str2double(x{1}), userIDs); 
[~, sortIdx] = sort(userIDs); 
files = files(sortIdx); 

variablesToAnalyze = {'Acc_FreqD_FDay', 'Acc_TimeD_FDay', 'Acc_FreqD_MDay', 'Acc_TimeD_MDay'};


allVariablesData = struct();
for i = 1:length(variablesToAnalyze)
    allVariablesData.(variablesToAnalyze{i}) = []; 
end

fprintf('=== STARTING INTER-VARIANCE ANALYSIS ===\n');


for fileIndex = 1:length(files)

    fileName = fullfile(dataFolder, files(fileIndex).name);
    data = load(fileName); 
    
 
    [~, fileBaseName, ~] = fileparts(files(fileIndex).name); 
    variableNameParts = split(fileBaseName, '_'); 
    variableName = strjoin(variableNameParts(2:end), '_'); 
    
    if ismember(variableName, variablesToAnalyze)
        userData = struct2cell(data); 
        allVariablesData.(variableName) = [allVariablesData.(variableName), userData{:}(:)];
    end
end

fprintf('\n=== Computing Inter-Variance Metrics and Plotting ===\n');
for i = 1:length(variablesToAnalyze)
    variableName = variablesToAnalyze{i};
    variableData = allVariablesData.(variableName);
    
    means = mean(variableData, 1); 
    variances = var(variableData, 1); 
    stdDevs = std(variableData, 1); 
    
    fprintf('Variable: %s\n', variableName);
    fprintf('Mean for Users: %s\n', mat2str(means, 3));
    fprintf('Variance for Users: %s\n', mat2str(variances, 3));
    fprintf('Std Dev for Users: %s\n\n', mat2str(stdDevs, 3));
    
  
    figure;
    userLabels = strcat('User', string(1:size(variableData, 2))); 
    statsMatrix = [means; variances; stdDevs]'; 
    
    bar(categorical(userLabels), statsMatrix);
    title(sprintf('Inter-Variance Analysis: %s', variableName));
    ylabel('Values');
    xlabel('Users');
    legend({'Mean', 'Variance', 'Std Dev'}, 'Location', 'northwest');
    grid on;
end

fprintf('=== INTER-VARIANCE ANALYSIS COMPLETE ===\n');
