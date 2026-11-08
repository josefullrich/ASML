
% EVALUATE_MODEL
% Offline evaluation of the surrogate models used by the Bayesian
% optimizer. A logged experiment is read from a csv file and the loop
% below repeatedly refits the EUV Gaussian process (GPMeuv) and the die
% yield model (GPMdy) on a growing training window, then qualifies each
% fit on the complete post-warmup data set. The resulting score curves
% show how many measurements are needed before the surrogate models are
% trustworthy.
%
%   training window     : warmup+init : i-1   (grows with i)
%   qualification set   : warmup+1    : n     (fixed, includes future rows)

% Measurement log to analyse and the folder holding fitcgp/ClassificationGP.
csv_file = 'opt_results_20261001_161410.csv';
this_dir = fileparts(mfilename('fullpath'));
srcmdl_path = this_dir;
if isfolder(srcmdl_path)
    addpath(srcmdl_path);
end

% Read the log. Each row is one measurement, in measurement order.
T = readtable(fullfile(fileparts(this_dir),'data',csv_file));

% The nine process settings in source-model input order
% (l2dx, l2dy, l2dz, pp energy, mp2pp delay, drive laser energy,
% feed-forward gain, beam size, dose target). The dose target must stay
% last because the EUV basis function uses the last column.
setting_vars = {'PP2DX_um','PP2DY_um','PP2DZ_um','PPE_mJ','MP2PPDelay_ns', ...
    'MMPED_mJ','FFWPED','MPBS_um','DoseTarget_mJ'};
missing_vars = setdiff([setting_vars {'EUV_mJ','NrOfExposures','NrOfInSpecDose', ...
    'Rx_deg','Ry_deg','TargetSize_um'}],T.Properties.VariableNames);
if ~isempty(missing_vars)
    error('evaluate_model:missingColumns','%s is missing column(s): %s', ...
        csv_file,strjoin(missing_vars,', '));
end

n = height(T); % number of rows in csv file
warmup = 20;
num_variables = numel(setting_vars); % number of process settings being optimized
init = 3*num_variables;

% storage
x_train = T{:,setting_vars};
euv_train = T.EUV_mJ;
rx_train = T.Rx_deg;
ry_train = T.Ry_deg;
ts_train = T.TargetSize_um;
exp_train = T.NrOfExposures;
dy_train = T.NrOfInSpecDose./T.NrOfExposures; % measured die yield [%/100]

% Rows without a usable yield measurement cannot train the binomial model.
if any(~isfinite(dy_train))
    error('evaluate_model:badYield','%d row(s) have NrOfExposures = 0.', ...
        sum(~isfinite(dy_train)));
end

% Only refit every eval_step-th measurement. Increase this to trade score
% resolution for run time; every fit trains two Gaussian processes.
eval_step = 1;

% A fit is called qualified once its die-yield score on the qualification
% set reaches this value [%].
r2_qualify_threshold = 90;

% Die-yield requirement. It is only used by the cost function below, which
% is kept so the objective can be plotted with the same definition as the
% optimizer uses.
lbdy = 0.995; % die yield lower bound

% Bayesian-optimization settings.
minpnts = num_variables*3; % wait for at least three samples per variable
basis = @(x) x(:,end); % euv and dose target have linear relationship
tol_dy = 1e-4;

% Select the model used to predict die yield:
%   0 = Generalized Linear Model, fitglm (binomial distribution, 2nd order polynomials)
%   1 = Regression Gaussian Process on logit(DY), fitrgp (logit-normal distribution, ardmatern52 kernel) 
%   2 = Classification Gaussian Process, fitcgp (binomial distribution, ardmatern52 kernel) 
dy_model = 2; 

% Select Rsquared score method used to detrmine prediction quality:
%   0 = SSE-based R2
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


warning('off','stats:classreg:learning:impl:GPImpl:GPImpl:SigmaMustBeGreaterThanSigmaLowerBound');
warning('off','stats:glmfit:IterationLimit');
warning('off','stats:LinearModel:RankDefDesignMat');
warning('off','ClassificationGP:CholeskyFailed');
warning('off','ClassificationGP:MaxIterReached');
warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:illConditionedMatrix');

% The qualification set is fixed: every measurement after the warmup,
% including the rows that the growing training window has not reached yet.
qual = warmup+1:n;

% Iterations at which the models are refitted and scored.
eval_iter = (warmup+init+1:eval_step:n)';
num_eval = numel(eval_iter);

% Score storage. NaN marks an iteration that was skipped or that failed
% to fit, so the plots simply show a gap instead of a misleading value.
train_size = NaN(num_eval,1);
r2_dy_train = NaN(num_eval,1);
r2_dy_qual = NaN(num_eval,1);
r2_euv_train = NaN(num_eval,1);
r2_euv_qual = NaN(num_eval,1);
rmse_euv_train = NaN(num_eval,1);
rmse_euv_qual = NaN(num_eval,1);
rmse_dy_qual = NaN(num_eval,1);

% Best qualified fit found so far, kept for the final parity plots.
best_r2 = -Inf;
best_i = NaN;
GPMeuv_best = [];
GPMdy_best = [];
dy_fit_best = [];
Rsquared = NaN;

% Refit the surrogate models on an ever larger part of the log.
for k = 1:num_eval
    i = eval_iter(k);

    % Training window: the measurements available at iteration i, with the
    % warmup and initial design rows forgotten.
    iter = warmup+1:i-1;
    train_size(k) = numel(iter);
    fprintf('Measurement %d of %d (training on %d points)\n',i,n,numel(iter))

    % A handful of points cannot support a nine-dimensional ARD kernel:
    % wait for at least three samples per variable (minpnts).
    if numel(iter) < minpnts
        fprintf('  skipped: fewer than %d training points\n',minpnts);
        continue
    end

    % Fit one Gaussian-process surrogate for each modelled response.
    % A Gaussian process supplies both a prediction and uncertainty,
    % which are needed to balance exploitation and exploration.
    % An ill-conditioned window can make a fit fail; such an iteration is
    % recorded as a missing score instead of aborting the whole sweep.
    try
        GPMeuv = fitrgp(x_train(iter,:),euv_train(iter),...
            'Basis',basis,'Beta',0.1,...
            'Standardize',1,'KernelFunction','ardmatern52',...
            'SigmaLowerBound',1e-2,'Sigma',1e-2,'ConstantSigma',true);
        if dy_model == 2
            GPMdy = fitcgp(x_train(iter,:),[dy_train(iter).*exp_train(iter) exp_train(iter)],...
                'Standardize',1,'Basis','none','KernelFunction','ardmatern52');
            dy_fit = @(xq) predict_dy_class(GPMdy,xq);
        elseif dy_model
            GPMdy = fitrgp(x_train(iter,:),logit(dy_train(iter)),...
                'Standardize',1,'Basis','none','KernelFunction','ardmatern52',...
                'SigmaLowerBound',tol_dy,'Sigma',tol_dy,'ConstantSigma',false);
            dy_fit = @(xq) sigmoid(predict(GPMdy,xq));
        else
            GPMdy = fitglm(x_train(iter,:),[dy_train(iter).*exp_train(iter) exp_train(iter)],'poly222222221',...
                'Distribution','binomial','Intercept',false,'Options',statset('MaxIter',1000));
            dy_fit = @(xq) predict(GPMdy,xq);
        end
    catch ME
        fprintf('  skipped: fit failed (%s)\n',ME.message);
        continue
    end

    % Predict on the training window and on the fixed qualification set.
    % The qualification set also contains measurements taken after i, so
    % the second score shows how well the fit extrapolates to the rest of
    % the experiment instead of how well it reproduces its own data.
    euv_pred_train = predict(GPMeuv,x_train(iter,:));
    euv_pred_qual = predict(GPMeuv,x_train(qual,:));
    dy_pred_train = dy_fit(x_train(iter,:));
    dy_pred_qual = dy_fit(x_train(qual,:));

    % Score both models. Observations with non-finite values are skipped
    % so that a single bad measurement does not turn a score into NaN.
    r2_dy_train(k) = dy_pseudo_r2(dy_actual_train(iter),dy_pred_train,r2_method,tol_dy);
    r2_dy_qual(k) = dy_pseudo_r2(dy_actual_train(qual),dy_pred_qual,r2_method,tol_dy);
    [r2_euv_train(k),rmse_euv_train(k)] = euv_score(euv_train(iter),euv_pred_train);
    [r2_euv_qual(k),rmse_euv_qual(k)] = euv_score(euv_train(qual),euv_pred_qual);
    rmse_dy_qual(k) = sqrt(mean((dy_pred_qual - dy_actual_train(qual)).^2,'omitnan'));
    Rsquared = r2_dy_qual(k);

    % Keep the best qualified fit for the final parity plots.
    if isfinite(Rsquared) && Rsquared > best_r2
        best_r2 = Rsquared;
        best_i = i;
        GPMeuv_best = GPMeuv;
        GPMdy_best = GPMdy;
        dy_fit_best = dy_fit;
    end

    fprintf(['  DY R^2: %7.3f%% train | %7.3f%% qual (RMSE %.4f) || ' ...
        'EUV R^2: %7.3f%% train | %7.3f%% qual (RMSE %.4f mJ)\n'], ...
        r2_dy_train(k),r2_dy_qual(k),rmse_dy_qual(k), ...
        r2_euv_train(k),r2_euv_qual(k),rmse_euv_qual(k));
end

%% Qualification summary
dy_model_name = {'fitglm (binomial)','fitrgp on logit(DY)','fitcgp (binomial)'};
r2_name = {'SSE-based R^2','deviance-based pseudo R^2','McFadden pseudo R^2'};

fprintf('\nQualification results for %s\n',csv_file);
fprintf('  Measurements        : %d (warmup %d, init %d)\n',n,warmup,init);
fprintf('  Training window     : %d:i-1 (%d to %d points)\n', ...
    warmup+init,min(train_size),max(train_size));
fprintf('  Qualification set   : %d:%d (%d points)\n',qual(1),qual(end),numel(qual));
fprintf('  DY model            : %s\n',dy_model_name{dy_model+1});
fprintf('  DY score            : %s\n',r2_name{r2_method+1});

if isfinite(best_i)
    fprintf('  Best DY qual. R^2   : %.3f%% at measurement %d\n',best_r2,best_i);
else
    fprintf('  Best DY qual. R^2   : not available (no successful fit)\n');
end

% First training size that reaches the qualification threshold, which is
% the practical answer to "how many measurements does the model need?".
qualified = find(r2_dy_qual >= r2_qualify_threshold,1,'first');
if isempty(qualified)
    fprintf('  DY R^2 >= %.1f%%     : never reached\n',r2_qualify_threshold);
else
    fprintf('  DY R^2 >= %.1f%%     : measurement %d (%d training points)\n', ...
        r2_qualify_threshold,eval_iter(qualified),train_size(qualified));
end

qualified_euv = find(r2_euv_qual >= r2_qualify_threshold,1,'first');
if isempty(qualified_euv)
    fprintf('  EUV R^2 >= %.1f%%    : never reached\n',r2_qualify_threshold);
else
    fprintf('  EUV R^2 >= %.1f%%    : measurement %d (%d training points)\n', ...
        r2_qualify_threshold,eval_iter(qualified_euv),train_size(qualified_euv));
end

%% Plots
line_width = 1.2;

figure('Name','Die-yield model score')
plot(train_size,r2_dy_train,'-','LineWidth',line_width,'DisplayName','Training window')
hold on
plot(train_size,r2_dy_qual,'-','LineWidth',line_width,'DisplayName','Qualification set')
yline(r2_qualify_threshold,'--','Qualification threshold','LineWidth',line_width)
grid on
title(sprintf('DY model (%s) score vs. training size',dy_model_name{dy_model+1}))
xlabel('Number of training measurements')
ylabel(sprintf('%s [%%]',r2_name{r2_method+1}))
ylim([-100 100])
legend('Location','best')
hold off

figure('Name','EUV model score')
plot(train_size,r2_euv_train,'-','LineWidth',line_width,'DisplayName','Training window')
hold on
plot(train_size,r2_euv_qual,'-','LineWidth',line_width,'DisplayName','Qualification set')
yline(r2_qualify_threshold,'--','Qualification threshold','LineWidth',line_width)
grid on
title('EUV Gaussian process score vs. training size')
xlabel('Number of training measurements')
ylabel('R^2 [%]')
ylim([-100 100])
legend('Location','best')
hold off

figure('Name','Prediction error')
plot(train_size,rmse_euv_qual,'-','LineWidth',line_width,'DisplayName','EUV RMSE [mJ]')
hold on
plot(train_size,rmse_dy_qual,'-','LineWidth',line_width,'DisplayName','DY RMSE [%/100]')
grid on
title('Qualification-set prediction error')
xlabel('Number of training measurements')
ylabel('Root-mean-square error')
legend('Location','best')
hold off

% Parity plots of the best qualified fit on the qualification set.
if isfinite(best_i)
    euv_meas = euv_train(qual);
    euv_hat = predict(GPMeuv_best,x_train(qual,:));
    dy_meas = dy_actual_train(qual);
    dy_hat = dy_fit_best(x_train(qual,:));

    figure('Name','Parity plots')
    subplot(1,2,1)
    plot(euv_meas,euv_hat,'*')
    hold on
    lims = [min([euv_meas;euv_hat]) max([euv_meas;euv_hat])];
    plot(lims,lims,'--','LineWidth',line_width)
    grid on, axis equal, xlim(lims), ylim(lims)
    title(sprintf('EUV, fit at measurement %d',best_i))
    xlabel('Measured EUV [mJ]')
    ylabel('Predicted EUV [mJ]')
    hold off

    subplot(1,2,2)
    plot(100.*dy_meas,100.*dy_hat,'*')
    hold on
    lims = 100.*[min([dy_meas;dy_hat]) max([dy_meas;dy_hat])];
    plot(lims,lims,'--','LineWidth',line_width)
    grid on, axis equal, xlim(lims), ylim(lims)
    title(sprintf('Die yield, fit at measurement %d',best_i))
    xlabel('Measured DY [%]')
    ylabel('Predicted DY [%]')
    hold off
end

%% Local functions

% Return the predicted probability of a passing die from the classification
% Gaussian process. A local function is used because an anonymous function
% cannot request the second output of predict.
function dy = predict_dy_class(model,xq)
    [~,scores] = predict(model,xq);
    dy = scores(:,2);
end

% Compare predicted and measured die yield with the selected pseudo
% R-squared definition [%]. Non-finite observations are dropped and the
% remaining values are clipped so that the logarithms stay finite.
function Rsquared = dy_pseudo_r2(dy_meas,dy_pred,r2_method,tol_dy)
    dy_meas = dy_meas(:);
    dy_pred = dy_pred(:);
    ok = isfinite(dy_pred) & isfinite(dy_meas);
    if ~any(ok)
        Rsquared = NaN;
        return
    end
    y = min(max(dy_meas(ok),tol_dy),1-tol_dy);
    yhat = min(max(dy_pred(ok),tol_dy),1-tol_dy);
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
end

% Classical R-squared [%] and root-mean-square error for a continuous
% response such as EUV.
function [Rsquared,rmse] = euv_score(y,yhat)
    y = y(:);
    yhat = yhat(:);
    ok = isfinite(y) & isfinite(yhat);
    if ~any(ok)
        Rsquared = NaN;
        rmse = NaN;
        return
    end
    y = y(ok);
    yhat = yhat(ok);
    SSR = sum((yhat - y).^2);
    TSS = sum((y - mean(y)).^2);
    Rsquared = 100*(1 - SSR/TSS);
    rmse = sqrt(mean((yhat - y).^2));
end

% x.*log(y) with the convention 0*log(0) = 0, matching the deviance and
% log-likelihood formulas above.
function v = xlogy(x,y)
    v = x.*log(y);
    v(x == 0) = 0;
end

