%% Clear Workspace and close all files, to run the new test %%
clear all; close all; clc;
load CCPP.dat
warning('off', 'fuzzy:general:warnGenfis1_Deprecation');
tic
%% Normalize the Data %%
[rows, columns] = size(CCPP);
format long
for count = 1: columns
    CCPP_max(count) = max(CCPP(:,count));
    CCPP_min(count) = min(CCPP(:,count));
end
for count = 1: columns
    CCPP(:,count) = (CCPP(:,count) - CCPP_min(count)) / (CCPP_max(count) - CCPP_min(count));
end
format short
%% Seperate dataset to D_trn, D_val, D_chk %%
%~ 1st 60% is the Data for Training, next 20% is the Data for Validating and last 20% is the Data for Chechking. ~%
TRN_persent = 0.6; VAL_persent = 0.2; CHK_persent = 0.2;
D_trn = CCPP(1:round(TRN_persent * rows),:);
D_val = CCPP(round(TRN_persent * rows)+1:round(TRN_persent * rows) + 1 + round(VAL_persent * rows),:);
D_chk = CCPP(round(TRN_persent * rows)+round(VAL_persent * rows)+2:end,:);
%% Create the model %%
%~ 3 Features, with Gaussian Membership Functions and Singleton Output ~%
Model = genfis1(D_trn, 3, 'gbellmf','constant'); %
%% Train the Model with anfis %%
Options = anfisOptions('InitialFIS', Model, 'ValidationData', D_val, 'EpochNumber', 250);
Options.DisplayANFISInformation = 0;
Options.DisplayErrorValues = 0;
Options.DisplayStepSize = 0;
Options.DisplayFinalResults = 0;
[TRN_FIS, TRN_Error, StepSize, CHK_FIS, CHK_Error] = anfis(D_trn, Options);
toc
%% Evaluate the model %%
Model_Output = evalfis(CHK_FIS, D_chk(:,1:4));
Model_Error = abs(D_chk(:,end) - Model_Output);
%% Compute the Metrics %%
MSE = sum(Model_Error.^2) / length(Model_Error);
RMSE = sqrt(MSE);

SS_Res = sum( (D_chk(:,end) - Model_Output) .^ 2);
SS_Tol = sum( (D_chk(:,end) - mean(D_chk(:,end))) .^ 2);
R_Squared = 1 - (SS_Res / SS_Tol);

NMSE = (sum((D_chk(:,end) - Model_Output) .^ 2) / length(Model_Output)) / var(D_chk(:,end)); 
NDEI = sqrt(NMSE);
fprintf('Mean Square Error (MSE): %f\n', MSE);
fprintf('Root Mean Square Error (RMSE): %f\n', RMSE);
fprintf('R^2: %f\n', R_Squared);
fprintf('Normalised Mean Square Error (NMSE): %f\n', NMSE);
fprintf('NDEI: %f\n', NDEI);
%% Plot Some Results %%
%~ Membership Functions of Initial Model ~%
figure('Name','Membership Functions of Initial Model','NumberTitle','off')
sgtitle('Model 2: Membership Functions of Initial Model')
subplot(2,2,1);
plotmf(Model,'input',1)
subplot(2,2,2);
plotmf(Model,'input',2)
subplot(2,2,3);
plotmf(Model,'input',3)
subplot(2,2,4);
plotmf(Model,'input',4)
%~ Membership Functions of Trained Model ~%
figure('Name','Membership Functions of Trained Model','NumberTitle','off')
sgtitle('Model 2: Membership Functions of Trained Model')
subplot(2,2,1);
plotmf(CHK_FIS,'input',1)
subplot(2,2,2);
plotmf(CHK_FIS,'input',2)
subplot(2,2,3);
plotmf(CHK_FIS,'input',3)
subplot(2,2,4);
plotmf(CHK_FIS,'input',4)
%~ Learning Curves ~%
figure('Name','Training and Checking Error for 3 Features','NumberTitle','off')
sgtitle(' Learning Curve of Model 2')
plot(TRN_Error); hold on
plot(CHK_Error)
xlabel('Epochs'); ylabel('Error');
xlim([0 250]); legend('Training Error', 'Checking Error');
%~ Prediction Error ~%
figure('Name','Prediction Error','NumberTitle','off')
sgtitle('Model 2: Prediction Error' )
subplot(1,2,1)
plot(Model_Output(1:100)); hold on;
plot(D_chk(1:100,end),'r'); legend(' Model Output ', ' Real Output ');
title('Model Output and Real Output')
xlim([1 length(Model_Output(1:100))]);
xlabel('index'); ylabel('Output');
subplot(1,2,2)
plot(Model_Error(1:100))
title('Prediction Error');
xlim([1 length(Model_Output(1:100))]);
xlabel('index'); ylabel('Error');
