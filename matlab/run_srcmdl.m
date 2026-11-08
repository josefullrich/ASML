% RUN_SRCMDL
% This script repeatedly runs the source model with one fixed set of
% process settings. It is a simple example of a closed-loop experiment:
% the die-yield measurement can be used to adjust the dose target before
% the next measurement.
%
% The settings below are normalized process inputs. A value of 0 or 1 is
% the lower or upper end of a setting's allowed range, respectively.
x = 0.250;
y = 0.700;
z = 0.477;
e = 0.798;
d = 0.150;
p = 0.275;
k = 0.312;
b = 0.657;

% Dose target per burst, in mJ. This is the input that the feedback
% controller changes during the experiment.
dt = 62.5;

% Die-yield (DY) feedback-controller settings.
% When enabled, the controller makes a small integral-style correction:
% if the measured DY is below the target, the dose target is reduced; if it
% is above the target, the dose target is increased.
enable_dy_control = 1;
integral_gain = 0.3;
dy_target = 0.995;

% Number of simulated measurements. The source model advances by about
% 10 seconds per measurement, so this run represents roughly 12 days.
n = 100000;
iter = 1:n;

% Preallocate arrays. Preallocation gives each output a fixed size and is
% much faster than growing arrays inside the loop.
dt_train = zeros(n,1);
euv_train = zeros(n,1);
rx_train = zeros(n,1);
ry_train = zeros(n,1);
ts_train = zeros(n,1);
dy_train = zeros(n,1);
euv_ol_train = zeros(n,1); % reference open-loop EUV
dy_actual_train = zeros(n,1); % actual yield before model averaging
state = 0; % source number 

% Run one source-model measurement at a time.
% 'state' is returned by srcmdl and passed into the next call so that the
% model can preserve its source history between measurements.
for i = 1:n
    [state,euv,rx,ry,ts,dy,nr_exposures,de,euv_ol,dy_actual] = srcmdl(x,y,z,e,d,p,k,b,dt,state);

    % Store the measurements for later plots. EUV can contain multiple
    % exposure values, so use its average for the displayed response.
    euv_train(i) = mean(euv);
    rx_train(i) = rx;
    ry_train(i) = ry;
    ts_train(i) = ts;
    dy_train(i) = dy;
    euv_ol_train(i) = euv_ol;
    dy_actual_train(i) = dy_actual;

    % Save the control input before changing it. This makes dt_train(i)
    % correspond to the dose target used for measurement i.
    dt_train(i) = dt;
    if enable_dy_control
        % Integral-style feedback update. The correction is proportional to
        % the difference between measured and desired die yield.
        dt = dt + integral_gain*(dy-dy_target);
    end
end

% Plot each response against elapsed time. There are 10 seconds per
% measurement, so multiplying the measurement index by 10 converts the
% horizontal axis from sample number to seconds.
time = 10.*iter;

figure, plot(time,euv_train)
grid on
xlabel('Elapsed time [s]')
ylabel('Average EUV [mJ]')
figure, plot(10.*iter,euv_ol_train)
grid on
xlabel('Elapsed time [s]')
ylabel('Open-loop EUV [mJ]')
figure, plot(10.*iter,rx_train)
grid on
xlabel('Elapsed time [s]')
ylabel('Average Rx [deg]')
figure, plot(10.*iter,ry_train)
grid on
xlabel('Elapsed time [s]')
ylabel('Average Ry [deg]')
figure, plot(10.*iter,ts_train)
grid on
xlabel('Elapsed time [s]')
ylabel('Average target size [um]')
figure, plot(10.*iter,100.*dy_train)
grid on
xlabel('Elapsed time [s]')
ylabel('Die yield [%]')
figure, plot(10.*iter,100.*dy_actual_train)
grid on
xlabel('Elapsed time [s]')
ylabel('Actual die yield [%]')

% Plot the dose-target control signal. This shows how feedback changes the
% input over time in an attempt to keep DY near dy_target.
figure, plot(time,dt_train)
grid on
xlabel('Elapsed time [s]')
ylabel('Dose target [mJ]')