
% This script demonstrates Bayesian optimization for the source model.
% The optimizer chooses nine process settings and learns which settings
% produce high EUV while satisfying target-shape and die-yield constraints.
%
% Bayesian optimization is useful when each measurement is expensive:
% instead of testing every possible setting, a statistical surrogate model
% predicts the response and an acquisition function chooses the next test.

% Number of measurements and the number of early measurements reserved for
% learning a useful initial model. One measurement represents 10 seconds.
n = 300;
warmup = 90;
state = 0; % source-model state
num_variables = 9; % number of process settings being optimized

% storage
euv_train = zeros(n,1);
rx_train = zeros(n,1);
ry_train = zeros(n,1);
ts_train = zeros(n,1);
dy_train = zeros(n,1);
exp_train = zeros(n,1);
euv_ol_train = zeros(n,1); % do not use
dy_actual_train = zeros(n,1);  % do not use

% Physical and quality constraints.
% "lb" and "ub" mean lower bound and upper bound, respectively. These
% values are also drawn on the result plots so that the constraints are
% visible when the script is used for teaching or debugging.
lbr = 50; % initial dose-target lower bound [mJ]
ubr = 60; % initial dose-target upper bound [mJ]
lbt = 0.1; % target size lower bound
ubt = 0.9; % target size upper bound
lbrx = -23.5; % target angle rx lower bound
ubrx = 23.5; % target angle rx upper bound
lbry = 11.5; % target angle ry lower bound
ubry = 23.5; % target angle ry upper bound
lbdy = 0.995; % die yield lower bound

% Bayesian-optimization settings.
minpnts = num_variables*3; % wait for at least three samples per variable
basis = @(x) x(:,end); % euv and dose target have linear relationship
sigma_ry = 0.5/3;
sigma_ts = 5/3;
tol_dy = 1e-4;
n_test = 50000;
epsilon = 0; % exploration factor

% Select the model used to predict die yield:
%   0 = Generalized Linear Model, fitglm (binomial distribution, 2nd order polynomials)
%   1 = Regression Gaussian Process on logit(DY), fitrgp (logit-normal distribution, ardmatern52 kernel) 
%   2 = Classification Gaussian Process, fitcgp (binomial distribution, ardmatern52 kernel) 
dy_model = 2; 

% Select Rsquared score method used to detrmine prediction quality:
%   0 = SSE-based R^2
%   1 = Deviance-based R^2
%   2 = Log-Likelyhood-based R^2
r2_method = 1; 

% Useful transforms. The logit maps probabilities to the whole real line;
% the sigmoid maps them back to the interval (0, 1). Clipping prevents
% numerical problems when a predicted yield is exactly 0 or 1.
logit   = @(p) log(max(p,tol_dy)./max(1-p,tol_dy));
sigmoid = @(x) min(max(1./(1+exp(-x)),-1+tol_dy),1-tol_dy);

% Cost (objective) function. Bayesian optimization maximizes this value.
% Higher EUV is desirable, while a yield below the required value receives
% a penalty. The logit-based penalty makes errors near 100% more visible.
cost_option = 2; % 0 - linear; 1 - multiplicative; 2 - logit
switch cost_option
    case 0
        multiplier = 10;
        costf = @(euv,dy) euv-multiplier*(abs(dy-lbdy));
    case 1
        multiplier = 2;
        costf = @(euv,dy) euv.*(1 - abs(dy-lbdy).^multiplier);
    case 2
        multiplier = 10;
        costf = @(euv,dy) euv-(abs(logit(dy)-logit(lbdy)))/multiplier;
end

% Latin hypercube sampling spreads the initial experiments across the
% nine-dimensional input space more evenly than purely random sampling.
x_train = lhsdesign(n,num_variables,"criterion","correlation");
x_train(:,end) = lbr + (ubr - lbr).*x_train(:,end); % start with low dose target to increase change for good DY

warning('off','stats:classreg:learning:impl:GPImpl:GPImpl:SigmaMustBeGreaterThanSigmaLowerBound');
warning('off','stats:glmfit:IterationLimit');
warning('off','stats:LinearModel:RankDefDesignMat');
warning('off','ClassificationGP:CholeskyFailed');
warning('off','ClassificationGP:MaxIterReached');
warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:illConditionedMatrix');

% These values are updated whenever the optimizer finds a feasible
% measured point. They are initialized so the final summary is still
% defined if the optimization model never starts.
best_dt = NaN;
best_dy = NaN;
best_x = NaN(1,num_variables);
Rsquared = NaN;

% Run the sequential experiment loop.
for i = 1:n
    fprintf('Measurement %d of %d\n',i,n)

    % Bayesian optimization starts after enough observations exist to fit
    % surrogate models. Before that point, the initial Latin hypercube
    % design supplies measurements without model-based decisions.
    if (i >= minpnts) && any(dy_train > 0.5)
        if i > minpnts+1
            start = min(warmup,max(i-minpnts,1)); % slowly forget initial warmup
        else
            start = 1;
        end
        iter = start:i-1;

        % Fit one Gaussian-process surrogate for each measured response.
        % A Gaussian process supplies both a prediction and uncertainty,
        % which are needed to balance exploitation and exploration.
        GPMeuv = fitrgp(x_train(iter,:),euv_train(iter),...
            'Basis',basis,'Beta',0.1,...
            'Standardize',1,'KernelFunction','ardmatern52',...
            'SigmaLowerBound',1e-2,'Sigma',1e-2,'ConstantSigma',true);
        GPMrx = fitrgp(x_train(iter,:),rx_train(iter),...
            'Standardize',1,'KernelFunction','ardmatern52',...
            'SigmaLowerBound',sigma_ry,'Sigma',sigma_ry,'ConstantSigma',false);
        GPMry = fitrgp(x_train(iter,:),ry_train(iter),...
            'Standardize',1,'KernelFunction','ardmatern52',...
            'SigmaLowerBound',sigma_ry,'Sigma',sigma_ry,'ConstantSigma',false);
        GPMts = fitrgp(x_train(iter,:),ts_train(iter),...
            'Standardize',1,'KernelFunction','ardmatern52',...
            'SigmaLowerBound',sigma_ts,'Sigma',sigma_ts,'ConstantSigma',false);
        if dy_model == 2
            GPMdy = fitcgp(x_train(iter,:),[dy_train(iter).*exp_train(iter) exp_train(iter)],...
                'Standardize',1,'Basis','none','KernelFunction','ardmatern52');
            dy_pred_train = predict_dy_class(GPMdy,x_train(iter,:));
        elseif dy_model
            GPMdy = fitrgp(x_train(iter,:),logit(dy_train(iter)),...
                'Standardize',1,'Basis','none','KernelFunction','ardmatern52',...
                'SigmaLowerBound',tol_dy,'Sigma',tol_dy,'ConstantSigma',false);
            dy_pred_train = sigmoid(predict(GPMdy,x_train(iter,:)));
        else
            GLMdy = fitglm(x_train(iter,:),[dy_train(iter).*exp_train(iter) exp_train(iter)],'poly222222221',...
                'Distribution','binomial','Intercept',false,'Options',statset('MaxIter',1000));
            dy_pred_train = predict(GLMdy,x_train(iter,:));
        end

        % These two quantities compare predicted and measured die yield.
        % They are printed as a simple teaching diagnostic for model fit.
        % Observations with non-finite values are skipped so that a single
        % bad measurement does not turn the diagnostic into NaN.
        dy_meas = dy_actual_train(iter,:);
        ok = isfinite(dy_pred_train) & isfinite(dy_meas);
        y = min(max(dy_meas(ok),tol_dy),1-tol_dy);
        yhat = min(max(dy_pred_train(ok),tol_dy),1-tol_dy);
        ynull = mean(y);
        switch r2_method
            case 0
                % Sum of squared residuals
                SSR = sum((yhat - y).^2);
                % Total sum of squares
                TSS = sum((y - ynull).^2);
                % R squared [%]
                Rsquared = 100*(1 - SSR/TSS);
            case 1
                % Deviance model
                Dmodel = 2 * sum( ...
                    xlogy(y,     y ./ yhat) + ...
                    xlogy(1-y, (1-y) ./ (1-yhat)));
                % Null model
                Dnull = 2 * sum( ...
                    xlogy(y,     y ./ ynull) + ...
                    xlogy(1-y, (1-y) ./ (1-ynull)));
                % Deviance-based pseudo R-squared [%]
                Rsquared = 100*(1 - Dmodel/Dnull);
            case 2
                % Binomial log-likelihood
                LLmodel = sum( ...
                    xlogy(y,     yhat) + ...
                    xlogy(1-y, 1-yhat));
                % Null model
                LLnull = sum( ...
                    xlogy(y,     ynull.*ones(size(y))) + ...
                    xlogy(1-y, (1-ynull).*ones(size(y))));
                % McFadden pseudo-R^2 [%]
                Rsquared = 100 * (1 - LLmodel/LLnull);
        end
       
        % Only previously measured points inside the target constraints may
        % define the best target used by the acquisition function.
        idxs1 = find((ts_train(iter,:) >= lbt) & (ts_train(iter,:) <= ubt) &...
            (ry_train(iter,:) >= lbry) & (ry_train(iter,:) <= ubry) &...
            (rx_train(iter,:) >= lbrx) & (rx_train(iter,:) <= ubrx));

        % Select the best feasible observed point as the current target.
        [target,idx1] = max(costf(euv_train(iter(idxs1),:),dy_pred_train(idxs1)));
        best_dt = x_train(iter(idxs1(idx1)),end);
        best_dy = dy_pred_train(idxs1(idx1),end);
        best_x = x_train(iter(idxs1(idx1)),:);

        % Generate candidate settings. This is another Latin hypercube,
        % concentrated around the best dose target found so far.
        x_test = lhsdesign(n_test,num_variables,"criterion","correlation");

        % sample around +/- 3% of best DT*DY so far
        if 0.98*best_dt*best_dy > lbr
            lbr1 = 0.97*best_dt*best_dy;
            ubr1 = 1.03*best_dt*best_dy;
        else
            lbr1 = lbr;
            ubr1 = ubr;
        end
        x_test(:,end) = lbr1 + (ubr1 - lbr1).*x_test(:,end);

        % Predict every candidate's response and its uncertainty.
        [euv_pred,euv_dev_pred,euv_bnd_pred] = predict(GPMeuv,x_test);
        rx_pred = predict(GPMrx,x_test);
        ry_pred = predict(GPMry,x_test);
        ts_pred = predict(GPMts,x_test);
        if dy_model == 2
            [~,dy_pred,dy_bnd_pred] = predict(GPMdy,x_test);
            dy_pred = dy_pred(:,2);
        elseif dy_model
            [dy_pred,~,dy_bnd_pred] = predict(GPMdy,x_test);
            dy_pred = sigmoid(dy_pred);
            dy_bnd_pred = sigmoid(dy_bnd_pred);
        else
            [dy_pred,dy_bnd_pred] = predict(GLMdy,x_test);
        end

        % Reject candidate settings whose predicted shape is outside the
        % required bounds.
        idxs = find((ts_pred >= lbt) & (ts_pred <= ubt) &...
            (ry_pred >= lbry) & (ry_pred <= ubry) &...
            (rx_pred >= lbrx) & (rx_pred <= ubrx));

        % Improvement over the current target. Positive values are useful.
        predictions = costf(euv_pred,dy_pred);
        I = predictions - target;

        % Estimate uncertainty in the objective. Large uncertainty makes a
        % candidate attractive because it may reveal a better setting.
        if any(isnan(dy_bnd_pred))
            Sigma = max(abs(costf(euv_bnd_pred(:,2),dy_pred) - costf(euv_bnd_pred(:,1),dy_pred))./4, tol_dy);
        else
            Sigma = max(abs(costf(euv_bnd_pred(:,2),dy_bnd_pred(:,2)) - costf(euv_bnd_pred(:,1),dy_bnd_pred(:,1)))./4, tol_dy);
        end

        % Standardized improvement: expected improvement relative to
        % uncertainty. epsilon controls the exploration penalty.
        Gamma = (I - epsilon.*Sigma)./Sigma;

        % Probability density and probability of improvement for a standard
        % normal variable. Together they form expected improvement.
        PT = normpdf(Gamma);

        % probability of improvement
        PI = normcdf(Gamma);

        % Expected improvement combines exploitation and exploration.
        exploitation = I.*PI;
        exploration = Sigma.*PT;
        EI = exploitation + exploration;

        % Evaluate the candidate with the largest expected improvement next.
        [~,idx] = max(EI(idxs));
        if ~isempty(idx)
            x_train(i,:) = x_test(idxs(idx),:);
        end
    end

    x = x_train(i,1);
    y = x_train(i,2);
    z = x_train(i,3);
    e = x_train(i,4);
    d = x_train(i,5);
    p = x_train(i,6);
    k = x_train(i,7);
    b = x_train(i,8);
    dt = x_train(i,9);

    % Run the expensive source-model measurement at the selected setting.
    [state,euv,rx,ry,ts,dy,nr_exposures,de,euv_ol,dy_actual] = srcmdl(x,y,z,e,d,p,k,b,dt,state);
    state.drift = zeros(6,1); % disable drift

    euv_train(i) = mean(euv);
    rx_train(i) = rx;
    ry_train(i) = ry;
    ts_train(i) = ts;
    dy_train(i) = dy;
    exp_train(i) = nr_exposures;
    euv_ol_train(i) = euv_ol;  % do not use, this value is now known in real
    dy_actual_train(i) = dy_actual;  % do not use, this value is now known in real

    % Report the best values currently available after every measurement.
    % During the initial measurements, the model has not been fitted yet,
    % so the diagnostics are displayed as NaN.
    fprintf('  Best DT: %.4f mJ | Best DY: %.6f (%.3f%%) | R^2 score: %.3f%%\n', ...
        best_dt,best_dy,100.*best_dy,Rsquared);
end

% Final DY predictions
if dy_model == 2
    dy_pred_train = predict_dy_class(GPMdy,x_train);
elseif dy_model
    dy_pred_train = sigmoid(predict(GPMdy,x_train));
else
    dy_pred_train = predict(GLMdy,x_train);
end

% Display the final optimization diagnostics. DY is reported both as a
% fraction and as a percentage so it can be compared directly with lbdy.
fprintf('\nFinal Bayesian-optimization results:\n');
fprintf('  Best feasible DT: %.4f mJ\n',best_dt);
fprintf('  Best feasible DY: %.6f (%.3f%%)\n',best_dy,100.*best_dy);
fprintf('  R^2 score: %.3f%%\n',Rsquared);

% Solve for the dose target that meets the required die yield exactly.
% The surrogate DY model is evaluated at the best feasible settings while
% only the dose target (last variable) is varied. fminbnd is a bounded
% one-dimensional minimizer, so the root DY(DT) = lbdy is found by
% minimizing the absolute distance between the predicted DY and lbdy.
if all(isfinite(best_x))
    if dy_model == 2
        dy_of_dt = @(dt) predict_dy_class(GPMdy,[best_x(1:end-1) dt]);
    elseif dy_model
        dy_of_dt = @(dt) sigmoid(predict(GPMdy,[best_x(1:end-1) dt]));
    else
        dy_of_dt = @(dt) predict(GLMdy,[best_x(1:end-1) dt]);
    end
    dt_objective = @(dt) abs(dy_of_dt(dt) - lbdy);
    options = optimset('TolX',1e-3);
    [dt_at_lbdy,dt_residual] = fminbnd(dt_objective,lbr,1.05*best_x(end),options);
    dy_at_dt = dy_of_dt(dt_at_lbdy);

    fprintf('\nDose target solved with fminbnd for DY = %.6f (%.3f%%):\n',lbdy,100.*lbdy);
    fprintf('  DT: %.4f mJ\n',dt_at_lbdy);
    fprintf('  Predicted DY at that DT: %.6f (%.3f%%)\n',dy_at_dt,100.*dy_at_dt);
    fprintf('  |DY - lbdy| residual: %.3e\n',dt_residual);
    if dt_residual > tol_dy
        fprintf('  Note: no DT inside [%.4f %.4f] mJ reaches the required DY.\n',lbr,ubr);
    end
else
    dt_at_lbdy = NaN;
    fprintf('\nDose target solved with fminbnd: not available (no feasible point found).\n');
end

% Plot results. Each point is one measurement; the vertical lines show the
% requirement that the optimizer is trying to satisfy. The time axis uses
% 10 seconds per measurement, matching the source-model example.
iter = 1:n;
time = 10.*iter;
plot_marker = '*';
line_width = 1.2;

figure('Name','EUV response')
plot(time,euv_train,plot_marker)
grid on
title('Closed-loop EUV response')
xlabel('Elapsed time [s]')
ylabel('Average EUV [mJ]')
xlim([0 10*n])
legend('Measured EUV','Location','best')

figure('Name','Open-loop EUV response')
plot(time,euv_ol_train,plot_marker)
grid on
title('Open-loop EUV response')
xlabel('Elapsed time [s]')
ylabel('Open-loop EUV [mJ]')
xlim([0 10*n])
legend('Open-loop EUV','Location','best')

figure('Name','Target angles')
plot(time,rx_train,plot_marker,'DisplayName','Rx')
hold on
plot(time,ry_train,plot_marker,'DisplayName','Ry')
yline(lbrx,'--','Rx lower bound','LineWidth',line_width,'DisplayName','Rx lower bound')
yline(ubrx,'--','Rx upper bound','LineWidth',line_width,'DisplayName','Rx upper bound')
yline(lbry,':','Ry lower bound','LineWidth',line_width,'DisplayName','Ry lower bound')
yline(ubry,':','Ry upper bound','LineWidth',line_width,'DisplayName','Ry upper bound')
grid on
title('Target angles and feasible bounds')
xlabel('Elapsed time [s]')
ylabel('Angle [deg]')
xlim([0 10*n])
ylim([min(lbrx,lbry)-2 max(ubrx,ubry)+2])
legend('Location','best')
hold off

figure('Name','Target size')
plot(time,ts_train,plot_marker,'DisplayName','Target size')
hold on
yline(lbt,'--','Lower bound','LineWidth',line_width)
yline(ubt,'--','Upper bound','LineWidth',line_width)
grid on
title('Target size and feasible bounds')
xlabel('Elapsed time [s]')
ylabel('Target size [um]')
xlim([0 10*n])
ylim([0 1])
legend('Location','best')
hold off

figure('Name','Die yield')
plot(time,100.*dy_train,plot_marker,'DisplayName','Observed DY')
hold on
plot(time,100.*dy_pred_train,plot_marker,'DisplayName','Predicted DY')
yline(100.*lbdy,'--','Required yield','LineWidth',line_width)
grid on
title('Die yield and required minimum')
xlabel('Elapsed time [s]')
ylabel('Die yield [%]')
xlim([0 10*n])
ylim([90 100])
legend('Location','best')
hold off

figure('Name','Actual die yield')
plot(time,100.*dy_actual_train,plot_marker,'DisplayName','Actual DY')
hold on
plot(time,100.*dy_pred_train,plot_marker,'DisplayName','Predicted DY')
yline(100.*lbdy,'--','Required yield','LineWidth',line_width)
grid on
title('Actual die yield and required minimum')
xlabel('Elapsed time [s]')
ylabel('Actual die yield [%]')
xlim([0 10*n])
ylim([90 100])
legend('Location','best')
hold off

figure('Name','Dose target')
plot(time,x_train(:,end),plot_marker,'DisplayName','Dose target')
hold on
yline(lbr,'--','Lower bound','LineWidth',line_width)
yline(ubr,'--','Upper bound','LineWidth',line_width)
grid on
title('Dose target selected by the optimizer')
xlabel('Elapsed time [s]')
ylabel('Dose target [mJ]')
xlim([0 10*n])
ylim([lbr-2 max(ubr,best_dt)+2])
legend('Location','best')
hold off

% Plot the objective used by the optimizer. This is not a physical loss;
% it is a single score that trades off EUV and die yield.
figure('Name','Optimization objective')
plot(time,costf(euv_train,dy_pred_train),plot_marker,'DisplayName','Objective')
grid on
title('Optimization objective over time')
xlabel('Elapsed time [s]')
ylabel('Objective value [EUV-equivalent units]')
xlim([0 10*n])
legend('Location','best')

% Return the predicted probability of a passing die from the classification
% Gaussian process. A local function is used because an anonymous function
% cannot request the second output of predict.
function dy = predict_dy_class(model,xq)
    [~,scores] = predict(model,xq);
    dy = scores(:,2);
end
