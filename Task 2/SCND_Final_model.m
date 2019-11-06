%% Clear Workspace and close all files, to run the new test %%
clear all; close all; clc;
load superconduct.dat
tic
%% Random Permutate the array, to shuffle the rows and run relieff to compute the Main Features%%
ranks = relieff(superconduct(:,1:end-1),superconduct(:,end),25);
fprintf('relieff is over.\n');
%% Seperate dataset to D_trn, D_val, D_chk %%
%~ 1st 60% is the Data for Training, next 20% is the Data for Validating and last 20% is the Data for Chechking. ~%
[rows, columns] = size(superconduct);
TRN_persent = 0.6; VAL_persent = 0.2; CHK_persent = 0.2;
D_trn = superconduct(1:round(TRN_persent * rows),:);
D_val = superconduct(round(TRN_persent * rows)+1:round(TRN_persent * rows) + 1 + round(VAL_persent * rows),:);
D_chk = superconduct(round(TRN_persent * rows)+round(VAL_persent * rows)+2:end,:);
%% Create the model Only with the Wanted Features %%
NF = 6; NR = 17;
opt = NaN(4,1);
opt(4) = 0;
fprintf('Initialize the Model.\n');
Init_Model = genfis3(D_trn(:,ranks(1:NF)), D_trn(:,end), 'sugeno', NR, opt);
%% Train the Model with anfis %%
Options = anfisOptions('InitialFIS', Init_Model, 'ValidationData', [D_val(:,ranks(1:NF)) D_val(:,end) ] , 'EpochNumber', 250);
Options.DisplayANFISInformation = 0;
Options.DisplayErrorValues = 0;
Options.DisplayStepSize = 0;
Options.DisplayFinalResults = 0;
fprintf('Starting tuning the Model...\n');
[TRN_FIS, TRN_Error, StepSize, CHK_FIS, CHK_Error] = anfis([D_trn(:,ranks(1:NF)) D_trn(:,end)], Options);
fprintf('Finish tuning the Model.\n');
%% Evaluate the model %%
Model_Output = evalfis(CHK_FIS, D_chk(:,ranks(1:NF)));
Model_Error = abs(D_chk(:,end) - Model_Output);
%% Compute the Metrics %%
save Optimal_TSK.mat
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
figure('Name','Some Membership Functions','NumberTitle','off')
sgtitle('Some Membership Functions')
subplot(2,2,1);
[xOut,yOut] = plotmf(Init_Model,'input',2);
[xOut2,yOut2] = plotmf(CHK_FIS,'input',2);
plot(xOut(:,3),yOut(:,3)); hold on
plot(xOut2(:,3),yOut2(:,3));
xlabel('MF3 @ Input 2'); ylabel('Degree of Membership');
legend('Initial Model', 'Tuned Model');
subplot(2,2,2);
[xOut,yOut] = plotmf(Init_Model,'input',3);
[xOut2,yOut2] = plotmf(CHK_FIS,'input',3);
plot(xOut(:,12),yOut(:,12)); hold on
plot(xOut2(:,12),yOut2(:,12));
xlabel('MF12 @ Input 3'); ylabel('Degree of Membership');
legend('Initial Model', 'Tuned Model');
subplot(2,2,3);
[xOut,yOut] = plotmf(Init_Model,'input',5);
[xOut2,yOut2] = plotmf(CHK_FIS,'input',5);
plot(xOut(:,3),yOut(:,3)); hold on
plot(xOut2(:,3),yOut2(:,3));
xlabel('MF3 @ Input 5'); ylabel('Degree of Membership');
legend('Initial Model', 'Tuned Model');
subplot(2,2,4);
[xOut,yOut] = plotmf(Init_Model,'input',6);
[xOut2,yOut2] = plotmf(CHK_FIS,'input',6);
plot(xOut(:,1),yOut(:,1)); hold on
plot(xOut2(:,1),yOut2(:,1));
xlabel('MF1 @ Input 6'); ylabel('Degree of Membership');
legend('Initial Model', 'Tuned Model');
%~ Predictions ~%
figure('Name','Predictions for Optimal Features','NumberTitle','off')
sgtitle(' Prediction and Real Values for Optimal Features' )
plot(D_chk(1:100,end),'bo')
hold on;
plot(Model_Output(1:100),'rx')
xlabel('Index'); ylabel('Output');
xlim([0 length(Model_Output(1:100))]);
legend('Real Values', 'Predictions');
%~ Learning Curves ~%
figure('Name','Training and Checking Error for Optimal Features','NumberTitle','off')
sgtitle(' Learning Curve of TSK Model for Optimal Features' )
plot(TRN_Error)
hold on;
plot(CHK_Error)
xlabel('Epochs'); ylabel('Error');
xlim([0 250]);
legend('Training Error', 'Checking Error');
%~ Prediction Error ~%
figure('Name','Prediction Error','NumberTitle','off')
sgtitle('Prediction Error of TSK Model for Optimal Features' )
subplot(1,2,1)
title('Model Output and Real Output')
plot(Model_Output(1:100)); hold on;
plot(D_chk(1:100,end),'r'); legend(' Tuned Model Output ', ' Real Output ');
xlim([1 length(Model_Output(1:100))]);
xlabel('Index'); ylabel('Output');
subplot(1,2,2)
title('Prediction Error');
plot(Model_Error(1:100))
xlim([1 length(Model_Output(1:100))]);
xlabel('Index'); ylabel('Error');
