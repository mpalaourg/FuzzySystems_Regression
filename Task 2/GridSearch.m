%% Clear Workspace and close all files, to run the new test %%
clear all; close all; clc;
load superconduct.dat 
%% Run relieff to compute the Main Features%%
ranks = relieff(superconduct(:,1:end-1),superconduct(:,end),25);
%% Pre-processing of data %%
%~ Check for NaN ~%
NaNs =  ismissing(superconduct(:));
if ~max(NaNs), fprintf('None NaN value was found in dataset.\n'); end
%~ Check for Infs ~%
Infs =  isinf(superconduct(:));
if ~max(Infs), fprintf('None Inf value was found in dataset.\n'); end
%% Seperate dataset to D_trn, D_val, D_chk %%
%~ 1st 60% is the Data for Training, next 20% is the Data for Validating and last 20% is the Data for Chechking. ~%
[rows, columns] = size(superconduct);
TRN_persent = 0.6; VAL_persent = 0.2; CHK_persent = 0.2;
D_trn = superconduct(1:round(TRN_persent * rows),:);
D_val = superconduct(round(TRN_persent * rows)+1:round(TRN_persent * rows) + 1 + round(VAL_persent * rows),:);
D_chk = superconduct(round(TRN_persent * rows)+round(VAL_persent * rows)+2:end,:);
%% Create Cross Validation Folds and Run the Tests%%
%~ For each value of NF and NR a model will be initialized by genfis3 and
%~ the Training Data will be seperated to 5 folds {80% Training - 20% Validating}
%~ by cvpartition. Each fold will be used to tuned anfis, for 100 Epochs 
%~ and the Min RMSE of the Epochs will be saved as the Fold Error. Finally,
%~ the Mean of this Error will be Computed as the metric to compute the
%~ Optimal Model.
fileID = fopen('ConsoleLog_Final_Test.txt', 'a');
NF = [3 6 9 12];
NR = [5 8 11 14 17];
Error = zeros(length(NF), length(NR));
opt = NaN(4,1);
opt(4) = 0;
for f = 1:length(NF)
    fprintf('Testing for # of Features: %d\n', NF(f));
    fprintf(fileID,'Testing for # of Features: %d\n', NF(f));
    for r = 1:length(NR)
        t_Start = tic;
        fprintf('\tTesting for # of Rules: %d', NR(r));
        fprintf(fileID,'\tTesting for # of Rules: %d', NR(r));
        Init_Model = genfis3(D_trn(:,ranks(1:NF(f))), D_trn(:,end), 'sugeno', NR(r), opt);
        c = cvpartition(D_trn(:,end), 'Kfold', 5);
        Curr_Error = 0;
        for count = 1:5
           training_idx = c.training(count);
           testing_idx  = c.test(count);
           
           Train_Set_In = D_trn(training_idx, ranks(1:NF(f)));
           Train_Set_Out = D_trn(training_idx, end);
           Validate_Set_In = D_trn(testing_idx, ranks(1:NF(f)));
           Validate_Set_Out = D_trn(testing_idx, end);
           
           Options = anfisOptions('InitialFIS', Init_Model, 'ValidationData', [Validate_Set_In Validate_Set_Out], 'EpochNumber', 100);
           Options.DisplayANFISInformation = 0;
           Options.DisplayErrorValues = 0;
           Options.DisplayStepSize = 0;
           Options.DisplayFinalResults = 0;
           
           [~, TRN_Error, ~, CHK_FIS, CHK_Error] = anfis([Train_Set_In Train_Set_Out], Options);
           NaNs =  ismissing(CHK_Error(:));
   %~ If a NaN value will be returned by anfis, then set the Fold Error to Inf and continue to the next NR(r) ~%
           if max(NaNs) 
               fprintf('\t Oooops NaN value was found @ NF = %d, NR = %d.', NF(f), NR(r));
               fprintf(fileID,'Oooops NaN value was found @ NF = %d, NR = %d.\n', NF(f), NR(r));
               Epoch_RMSE = Inf;
               Curr_Error = Curr_Error + Epoch_RMSE;
               break;
           else
               Epoch_RMSE = min(CHK_Error(:));
               Curr_Error = Curr_Error + Epoch_RMSE;
           end
        end
        t_End = toc(t_Start);
        fprintf('\t Elapsed Time: %f sec.\n', t_End);
        fprintf(fileID,'\t Elapsed Time: %f sec.\n', t_End);
        Error(f, r) = Curr_Error / 5;
    end
    save Final_Test.mat
end
fclose(fileID);
%% Plots The Error of the Grid Search at 2-D Bar and 3D %%
Optimal_Value = min(min(Error));
[Opt_x,Opt_y] = find(Error == Optimal_Value);
fprintf('The Optimal Model is for NF = %d, NR = %d.\n', NF(Opt_x), NR(Opt_y));
%~ Bar Plot 2D ~%
figure('Name','Bar Plot 2D','NumberTitle','off')
sgtitle('Bar Plot Error - 2D')
subplot(2,2,1);
bar(Error(1,:))
xlabel('Number of Rules');
ylabel('Mean RMSE');
xticklabels(string(NR));
legend([num2str(NF(1)),' features'])
subplot(2,2,2);
bar(Error(2,:))
xlabel('Number of Rules');
ylabel('Mean RMSE');
xticklabels(string(NR));
legend([num2str(NF(2)),' features'])

subplot(2,2,3);
bar(Error(3,:))
xlabel('Number of Rules');
ylabel('Mean RMSE');
xticklabels(string(NR));
legend([num2str(NF(3)),' features'])

subplot(2,2,4);
bar(Error(4,:))
xlabel('Number of Rules');
ylabel('Mean RMSE');
xticklabels(string(NR));
legend([num2str(NF(4)),' features'])
%~ Bar Plot 3D ~%
figure('Name','Bar Plot 3D','NumberTitle','off')
b = bar3(Error);
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end
xlabel('Number of Rules');
ylabel('Number of Features');
zlabel('Mean RMSE');
xticklabels(string(NR));
yticklabels(string(NF));
