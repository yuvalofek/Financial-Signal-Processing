%%% Yuval Epstain Ofek
%%% FSP PS4
%% 1
clear all;close all;clc;

%%% 1. 
%Params
N = 1e6;
alpha = 0.544;

%Useful functions
theo_var = @(v) v/(v-2);
over_4 = @(vals) mean(abs(vals)>4);

%Generate distr. 
gauss = randn(N,1);
cauchy = alpha*tan(pi*rand(N, 1));
t5 = trnd(5,N,1)/sqrt(theo_var(5));
t10 = trnd(10,N,1)/sqrt(theo_var(10));

%Find the fractions
frac_gauss = over_4(gauss)
frac_cauchy = over_4(cauchy)
frac_t5 = over_4(t5)
frac_t10 = over_4(t10)

figure
plot(cauchy)
xlabel('Index')
ylabel('Value')
title('Plot of Cauchy Data')
yl = ylim;
ylim([yl(1)*0.2, yl(2)*0.2])
%We see some really high values, but with low prob. 

%% 2
clear all; 

N = 250;
z = [0.2, 0.5];
p = [-0.8, -0.7];
M =10;

%Useful quick functions
to_poly = @(v) [1, v(1)+v(2), v(1)*v(2)];

%Generating r
a = to_poly(z);
b = to_poly(p);

%Inititalize, first length(z) are before t = 0. 
v = randn(N+length(z),1);
r = zeros(N+length(z),1);
v(1:length(z)) = zeros(length(z), 1);
for i = 1:N
    %Forward propogate to be able to start loop at 1
    % z^-2 -> 1
    % ==> 1 -> z^2
    r(i+2) = -b(2)*r(i+1)-b(3)*r(i) + v(i+2) + a(2)*v(i+1)+a(3)*v(i);
end
r = r(end-N+1:end);

%%% I made a function that prints all of the useful values (see bottom)
AR_analysis(r, M, N)
% Eigenvalues are positive real :)

% We have |k|<1 for all k_i's. This means that Pm>0 and the process is not
% entirely predictable.

%% Close to unit root nonstationarity
%%%3. Here we add another z^-1 to the denominator. 
p3 = [1, -0.99];
b3 = conv(b,p3);

% Wasn't sure if to use the same v values or not so I tried both. There
% wasn't much of a difference, but I felt like the plot comparing the two
% r_t values ended up nicer when I kept the same v, so I ended up keeping
% the v from the previous problem (2). 

%v = randn(N+length(b3)-1,1);
v = [0;v];   %To account for the additional delay. 

r3 = zeros(N+length(b3)-1,1);
for i = 1:N
    %Forward propogate to be able to start loop at 1
    % z^-3 -> 1
    % ==> 1 -> z^3
    r3(i+3) = -b3(2)*r3(i+2)-b3(3)*r3(i+1)-b3(4)*r3(i)+ v(i+3)+ a(2)*v(i+2)+a(3)*v(i+1);
end
r3 = r3(end-N+1:end);
figure
plot(r)
hold on 
plot(r3)
xlabel('t')
ylabel('value')
legend('ARMA(2,2) r_t', 'ARIMA(2,1,2) r_t with (close to) unit root nonstationarity', ...
    'Location', 'southwest' )
title('Comparing ARMA(2,2) and ARIMA(2,1,2) generated r_t''s')
AR_analysis(r3, M, N)
% E-vals are still positive real, but much larger than before. 
% Again we don't have perfect predictability, but we are much closer than
% before. 
%%
s = r3(2:end)-r3(1:end-1);
AR_analysis(s, M, N-1)

%%%% Comments:
% We see that in our plot of the (\gamma(m) / \gamma(0))s when we use r by
% itself we have a lot of high values (to the point  where they don't go
% under 0.8 in certain cases. This isn't really good for the analysis. Once
% we take the first difference, we notice that the values drop a lot
% 'nicer' in a sense that they don't all have around the same value. The
% results of this are also followed with much smaller eigenvalues.

% If we were tasked with detecting the unit root nonstationarity, we can
% look at these plots and if the values are around the same high value we
% know there is a (close to) unit root nonstationarity. 
%% 4. repeating 2, but with student t-distr instead of N(0,1)
clear all
N = 250;
z = [0.2, 0.5];
p = [-0.8, -0.7];
M =10;

%Useful quick functions
to_poly = @(v) [1, v(1)+v(2), v(1)*v(2)];

%Generating r
a = to_poly(z);
b = to_poly(p);

theo_var = @(v) v/(v-2);

v = trnd(5,N+length(b)-1,1)/sqrt(theo_var(5));

r = zeros(N+length(b)-1,1);
v(1:length(z)) = zeros(length(z), 1);
for i = 1:N
    %Forward propogate to be able to start loop at 1
    % z^-2 -> 1
    % ==> 1 -> z^2
    r(i+2) = -b(2)*r(i+1)-b(3)*r(i) + v(i+2) + a(2)*v(i+1)+a(3)*v(i);
end
r = r(end-N+1:end);


AR_analysis(r, M, N)
% Eigenvalues are positive real again:)
% Model is stable. 
% We have |k|<1 for all k_i's. This means that Pm>0 and the process is not
% entirely predictable. Value close to what we saw earlier. 

%% repeating 3, but with student t-distr instead of N(0,1)
p3 = [1, -0.99];
b3 = conv(b,p3);

% Wasn't sure if to use the same v values or not so I tried both. There
% wasn't much of a difference, but I felt like the plot comparing the two
% r_t values ended up nicer when I kept the same v, so I ended up keeping
% the v from the previous problem (2). 

%v = randn(N+length(b3)-1,1);
v = [0;v];   %To account for the additional delay. 

r3 = zeros(N+length(b3)-1,1);
for i = 1:N
    %Forward propogate to be able to start loop at 1
    % z^-3 -> 1
    % ==> 1 -> z^3
    r3(i+3) = -b3(2)*r3(i+2)-b3(3)*r3(i+1)-b3(4)*r3(i)+ v(i+3)+ a(2)*v(i+2)+a(3)*v(i+1);
end
r3 = r3(end-N+1:end);
figure
plot(r)
hold on 
plot(r3)
xlabel('t')
ylabel('value')
legend('ARMA(2,2) r_t', 'ARIMA(2,1,2) r_t with (close to) unit root nonstationarity', ...
    'Location', 'southwest' )
title('Comparing ARMA(2,2) and ARIMA(2,1,2) generated r_t''s')
AR_analysis(r3, M, N)
% E-vals are still positive real, but much larger than before, as we saw in
% problem 3. 
% Model is not as stable. 
% Again we don't have perfect predictability, but we are much closer than
% before. 
% I didn't see any stark differences between the two v distributions. 
%%
s = r3(2:end)-r3(1:end-1);
AR_analysis(s, M, N-1)

% Once again taking the first difference fixed the problems we saw with the
% unit root nonstationarity, and it affected the values in essentially the
% same ways. 

%% 5. 
clear all;close all;clc
N = 250;

%Problem Parameters
z = [0.2, 0.3];
p = [-0.2, -0.5];
delta =  [0.3;0.5];
G = [0.99, 0; ...
     0.3, 0.3];

A = [3, 4;...
     2, 3];
%Useful quick functions
to_poly = @(v) [1, v(1)+v(2), v(1)*v(2)];

%Generating 
a = to_poly(z);
b = to_poly(p);

% a. 
delta_p = A*delta
G_p = A*G*inv(A)

%%% Repeating 3 times: 
for jj = 1:3
    %b. we see that a_t = delta(1) + 0.99*a_t-1 + u_t.
    %Generate u_t:
    u_gen = [zeros(length(b)-1,1);randn(N,1)];
    u = zeros(N+length(b)-1,1);
    for i = 1:N
        u(i+2) = -b(2)*u(i+1)-b(3)*u(i) + u_gen(i+2) + a(2)*u_gen(i+1)+a(3)*u_gen(i);
    end
    u = u(end-N+1:end);    

    %Therefore a:
    a = zeros(N+1,1);
    for i = 1:N
        a(i+1) = delta(1) + G(1,1)*a(i) + u(i);
    end
    sig2_0 = mean(a.^2)/16

    %Generating the v
    theo_var = @(v) v/(v-2);

    sig2_0 = 1;

    %Gaussian
    v_g = sqrt(sig2_0)*randn(N,1);
    %Student t, v = 5
    v_t = trnd(5,N,1)/sqrt(theo_var(5)/sig2_0);

    %Then b: 
    b_g = zeros(N+1,1);
    b_t = zeros(N+1,1);
    for i = 1:N
        b_g(i+1) = delta(2) + G(2,1)*a(i) + G(2,2)*b_g(i) + v_g(i);
        b_t(i+1) = delta(2) + G(2,1)*a(i) + G(2,2)*b_t(i) + v_t(i);
    end

    r_g = A*[a.';b_g.'];
    r_t = A*[a.';b_t.'];


    %c. 
    figure
    subplot(3,1,1)
    plot(u)
    hold on
    plot(v_g)
    plot(v_t)
    xlabel('t')
    ylabel('value')
    legend('u_t', 'v_t - Gaussian', 'v_t - Student t')
    title('Graph of u_t and v_t''s')
    xlim([0,N])


    subplot(3,1,2)
    plot(a(2:end))
    hold on
    plot(b_g(2:end))
    plot(b_t(2:end))
    xlabel('t')
    ylabel('value')
    legend('a_t', 'b_t - Gaussian', 'b_t - Student t',...
         'Location', 'southeast')
    title('Graph of a_t and b_t''s')
    xlim([0,N])


    subplot(3,1,3)
    for i = 1:2
        plot(r_g(i,:))
        hold on
        plot(r_t(i,:))
    end
    xlabel('t')
    ylabel('value')
    legend('t_{1t} - Gaussian', 't_{1t} - Student t', 't_{2t} - Gaussian', ...
        't_{2t} - Student t', 'Location', 'southeast')
    title('Graph of r_t''s')
    xlim([0,N])
end

%d. 
coint = inv(A);
%a_t = coint(1,1)*r_{1t} + coint(1,2)*r_{2t}
%b_t = coint(2,1)*r_{1t} + coint(2,2)*r_{2t}

%e. 
%Autocorrelations
%Gaussian 
figure 
for i = 1:2    
    subplot(2,2,i)
    [acf,lags,bounds,h] = autocorr(r_g(i,:));
    title(['r_{',num2str(i),'t} Autocorrelation - Gaussian'])
end
%Student t
for i = 1:2    
    subplot(2,2,i+2)
    [acf,lags,bounds,h] = autocorr(r_t(i,:));
    title(['r_{',num2str(i),'t} Autocorrelation - Student t'])
end
%Cross correlations
figure 
subplot(2,1,1)
crosscorr(r_g(1,:), r_g(2,:))
title('r_t cross correlation - for Gaussian')

subplot(2,1,2)
crosscorr(r_t(1,:), r_t(2,:))
title('r_t cross correlation - for Student t')

%f. 
s_g = r_g(:,2:end)-r_g(:,1:end-1);
s_t = r_t(:,2:end)-r_t(:,1:end-1);

 
%Gaussian 
figure 
for i = 1:2    
    subplot(2,2,i)
    autocorr(s_g(i,:));
    title(['s_{',num2str(i),'t} Autocorrelation - Gaussian'])
end
%Student t
for i = 1:2    
    subplot(2,2,i+2)
    autocorr(s_t(i,:));
    title(['s_{',num2str(i),'t} Autocorrelation - Student t'])
end
%%% As expected, we see a much steeper decay. 
%Cross correlations
figure 
subplot(2,1,1)
crosscorr(s_g(1,:), s_g(2,:))

title('s_t cross correlation - for Gaussian')

subplot(2,1,2)
crosscorr(s_t(1,:), s_t(2,:))
title('s_t cross correlation - for Student t')

%g. 
%Generate c
xi = randn(N+1,1);
c = zeros(N+1,1);
for i = 1:N
   c(i+1) = 0.99*c(i) + xi(i);
end
c = c(end-N+1:end);

figure
for i = 1:2    
    subplot(2,2,i)
    crosscorr(c,r_g(i,:));
    title([' c & r_{',num2str(i),'t} Crosscorrelation - Gaussian'])
end
%Student t
for i = 1:2    
    subplot(2,2,i+2)
    crosscorr(c,r_t(i,:));
    title(['c &r_{',num2str(i),'t} Crosscorrelation - Student t'])
end

%%% As expected, we see much less crosscorrelation and changes every time I
%%% run the code

%% 6. 
%clear all;close all;clc;
%Import all as numeric matrix
uiopen('C:\Users\yuval\Documents\MATLAB\^SP500TR (1).csv',1)
uiopen('C:\Users\yuval\Documents\MATLAB\^SP500TR (2).csv',1)
uiopen('C:\Users\yuval\Documents\MATLAB\AAPL (1).csv',1)
uiopen('C:\Users\yuval\Documents\MATLAB\AAPL (2).csv',1)
uiopen('C:\Users\yuval\Documents\MATLAB\ADBE (1).csv',1)
uiopen('C:\Users\yuval\Documents\MATLAB\ADBE (2).csv',1)

%%
%useful Functions
lret = @(S) log(S(2:end)./S(1:end-1));

theo_var = @(v) v/(v-2);

%%% Organizing the financial data
Snp2018 = SP500TR1(:, end-1);
Snp2019 = SP500TR2(:, end-1);
APL18 = AAPL1(:, end-1);
APL19 = AAPL2(:, end-1);
ADBE18 = ADBE1(:, end-1);
ADBE19 = ADBE2(:, end-1);

R_18 = 1000*lret(Snp2018);
R_19 = 1000*lret(Snp2019);
SR_18 = R_18.^2/1000;
SR_19 = R_19.^2/1000;

AR_18 = 1000*lret(APL18);
AR_19 = 1000*lret(APL19);
SAR_18 = AR_18.^2/1000;
SAR_19 = AR_19.^2/1000;

ADR_18 = 1000*lret(ADBE18);
ADR_19 = 1000*lret(ADBE19);
SADR_18 = ADR_18.^2/1000;
SADR_19 = ADR_19.^2/1000;

%Plotting returns and square returns
figure
subplot(2,3,1)
plot(R_18)
hold on 
plot(SR_18)
xlim([0, length(SR_18)])
xlabel('Time')
ylabel('Value (in 10^{-3})')
legend('Returns', 'Square Returns')
title('S&P500 2018 Returns')

subplot(2,3,2)
plot(AR_18)
hold on 
plot(SAR_18)
xlim([0, length(SAR_18)])
xlabel('Time')
ylabel('Value (in 10^{-3})')
legend('Returns', 'Square Returns')
title('Apple 2018 Returns')


subplot(2,3,3)
plot(ADR_18)
hold on 
plot(SADR_18)
xlim([0, length(SADR_18)])
xlabel('Time')
ylabel('Value (in 10^{-3})')
legend('Returns', 'Square Returns')
title('Adobe 2017 Returns')
legend('Returns', 'Square Returns')

subplot(2,3,4)
plot(R_19)
hold on 
plot(SR_19)
xlim([0, length(SR_19)])
title('S&P500 2019 Returns')
xlabel('Time')
ylabel('Value (in 10^{-3})')
legend('Returns', 'Square Returns')

subplot(2,3,5)
plot(AR_19)
hold on 
plot(SAR_19)
xlim([0, length(SAR_19)])
title('Apple 2019 Returns')
xlabel('Time')
ylabel('Value (in 10^{-3})')
legend('Returns', 'Square Returns')

subplot(2,3,6)
plot(ADR_19)
hold on 
plot(SADR_19)
xlim([0, length(SADR_19)])
title('Adobe 2019 Returns')
xlabel('Time')
ylabel('Value (in 10^{-3})')
legend('Returns', 'Square Returns')

%Subtract sample means
R_18 = R_18 - mean(R_18);
R_19 = R_19 - mean(R_19);
SR_18 = SR_18 - mean(SR_18);
SR_19 = SR_19 - mean(SR_19);
AR_18 = AR_18 - mean(AR_18);
AR_19 = AR_19 - mean(AR_19);
SAR_18 = SAR_18 - mean(SAR_18);
SAR_19 = SAR_19 - mean(SAR_19);
ADR_18 = ADR_18 - mean(ADR_18);
ADR_19 = ADR_19 - mean(ADR_19);
SADR_18 = SADR_18 - mean(SADR_18);
SADR_19 = SADR_19 - mean(SADR_19);


%Checking Autocorrelations: 
%S&P500
figure
sgtitle('S&P500')
subplot(2,2,1)
autocorr(R_18);
title('Autocorrelation for Returns - 2018')
subplot(2,2,3)
autocorr(SR_18);
title('Autocorrelation for Square Returns - 2018')

subplot(2,2,2)
autocorr(R_19);
title('Autocorrelation for Returns - 2019')
subplot(2,2,4)
autocorr(SR_19);
title('Autocorrelation for Square Returns - 2019')

%APPLE
figure
sgtitle('Apple')
subplot(2,2,1)
autocorr(AR_18);
title('Autocorrelation for Returns - 2018')
subplot(2,2,3)
autocorr(SAR_18);
title('Autocorrelation for Square Returns - 2018')

subplot(2,2,2)
autocorr(AR_19);
title('Autocorrelation for Returns - 2019')
subplot(2,2,4)
autocorr(SAR_19);
title('Autocorrelation for Square Returns - 2019')

%Adobe
figure
sgtitle('Adobe')
subplot(2,2,1)
autocorr(ADR_18);
title('Autocorrelation for Returns - 2018')
subplot(2,2,3)
autocorr(SADR_18);
title('Autocorrelation for Square Returns - 2018')

subplot(2,2,2)
autocorr(ADR_19);
title('Autocorrelation for Returns - 2019')
subplot(2,2,4)
autocorr(SADR_19);
title('Autocorrelation for Square Returns - 2019')
%All look good

%%
%a. 
N =250;
w = 0.5;
alpha = 0.3;
beta = 0.4;

%Gaussian
z = [0;randn(N,1)];
[r_g, sigma_g] = synthesizeGARCH(z, w, alpha, beta);

figure
subplot (2,1,1)
plot(r_g)
hold on
plot(sigma_g)
xlabel('t')
ylabel('value')
title('Superimposed r_t and \sigma_t, Gaussian z_t')
legend('r_t', '\sigma_t')
xlim([0,N])

%Student t
st = [0; trnd(5,N,1)/sqrt(theo_var(5))];
[r_s, sigma_s] = synthesizeGARCH(st, w, alpha, beta);

subplot (2,1,2)
plot(r_s)
hold on
plot(sigma_s)
xlabel('t')
ylabel('value')
title('Superimposed r_t and \sigma_t, Student-t z_t, \nu = 5')
legend('r_t', '\sigma_t')
xlim([0,N])

%%% Gaussian
% Estimating GARCH
Mdl = garch(1,1);
EstGARCH = estimate(Mdl, r_g)
%We get values close to, but not equal to the values we were expecting (I
%saw fluctuations in all paramaters by around 0.2. 

% Estimating ARCH
Mdl = garch(0,2);
EstARCH = estimate(Mdl, r_g);

sig_h = simulate(EstARCH, N);
r_h = z(2:end).*sig_h;

figure
plot(r_h)
hold on
plot(sig_h)
xlabel('t')
ylabel('value')
title('Superimposed r_t and \sigma_t, as generated by ARCH model (Gaussian)')
legend('r_t', '\sigma_t')
xlim([0,N])
% Comments:
% Yes, we have a good match for the volitility. 


%%% Student t
% Estimating GARCH
Mdl = garch(1,1);
EstGARCH = estimate(Mdl, r_s)
%We get values close to, but not equal to the values we were expecting (I
%saw fluctuations in all paramaters by around 0.2. 

% Estimating ARCH
Mdl = garch(0,2);
EstARCH = estimate(Mdl, r_s);

sig_h = simulate(EstARCH, N);
r_h = st(2:end).*sig_h;

figure
plot(r_h)
hold on
plot(sig_h)
xlabel('time')
ylabel('Value')
title('Superimposed r_t and \sigma_t, as generated by ARCH model (Student t)')
legend('r_t', '\sigma_t')
xlim([0,N])
% Comments:
% student t-distribution didn't have too much of a difference on the
% parameter values, as they were still close to the values we used to
% generate the data. This time, however, the volitility wasn't as good of a
% match for the r_t anymore. 

%%
%%% b. Financial data

figure
sgtitle('S&P500')
subplot(2,2,1)
runestimate(R_18, 'r_t and  infered \sigma_t - 2018 Returns')
subplot(2,2,2)
runestimate(R_19, 'r_t and  infered \sigma_t - 2019 Returns')
subplot(2,2,3)
runestimate(SR_18, 'r_t and  infered \sigma_t - 2018 Square Returns')
subplot(2,2,4)
runestimate(SR_19, 'r_t and  infered \sigma_t - 2019 Square Returns')

figure
sgtitle('Apple')
subplot(2,2,1)
runestimate(AR_18, 'r_t and  infered \sigma_t - 2018 Returns')
subplot(2,2,2)
runestimate(AR_19, 'r_t and  infered \sigma_t - 2019 Returns')
subplot(2,2,3)
%Had lower bound errors, so I increased the scale:
runestimate(SAR_18*10000, 'r_t and  infered \sigma_t - 2018 Square Returns')
ylabel('value (in 10^{-8})')
subplot(2,2,4)
runestimate(SAR_19, 'r_t and  infered \sigma_t - 2019 Square Returns')

figure
sgtitle('Adobe')
subplot(2,2,1)
runestimate(ADR_18, 'r_t and  infered \sigma_t - 2018 Returns')
subplot(2,2,2)
runestimate(ADR_19, 'r_t and  infered \sigma_t - 2019 Returns')
subplot(2,2,3)
runestimate(SADR_18*1e3, 'r_t and  infered \sigma_t - 2018 Square Returns')
subplot(2,2,4)
runestimate(SADR_19, 'r_t and  infered \sigma_t - 2019 Square Returns')

%%% Comments:
% The values we get for the volitility seem to fit the data pretty well
% for all cases (ARCH vs. GARCH and square returns vs.% returns),
% which we see by how the variances were almost like an envelope 
% to the returns. The metrics for the estimates also showed this to be the 
% case. Note that in some instances the square returns were initially too 
% small and I had to rescale it. 

%% Functions
function runestimate(data, ttle)
N = length(data);
EstGARCH = estimate(garch(1,1),data);
EstARCH = estimate(garch(0,2), data);

v_garch = infer(EstGARCH, data);
v_arch = infer(EstARCH, data);

plot(data)
hold on
plot(sqrt(v_garch))
plot(sqrt(v_arch))
xlabel('t')
ylabel('value (in 10^{-3})')
title(ttle)
legend('r_t', '\sigma_t - GARCH', '\sigma_t - ARCH')
xlim([0,N])
end

function [r, sigma] = synthesizeGARCH(z, w, alpha, beta)
N = length(z);
sigma = zeros(N,1);    
r = zeros(N,1);
for i = 1:N-1
    sigma(i+1) = sqrt(w+alpha*r(i)^2+beta*sigma(i)^2);
    r(i+1) = sigma(i+1)*z(i+1);
end

end

function AR_analysis(r, M, N)
%a. 
figure
[acf,~,~,~] = autocorr(r, 'NumLags', M);

%b. 
gamma_0 = var(r);
C = toeplitz(acf*gamma_0);
evals = sort(eig(C), 'descend')

%c. 
L_chol = chol(C,'lower');
D = diag(diag(L_chol).^2);
L = (L_chol*(D^-0.5));

%d. 
P = zeros(M, 1);
F = zeros(M+1);
for m = 1:M
    [a,P(m),k] = levinson(C(:,1), m);
    F(end-(m):end, end-m) = a;
end
F(end,end) = 1;
%e. 
FCFT = F*C*F.';

%Check the diagonal of D and the values of P are REALLY close.
L1_P_D_distance = max(P - diag(D(2:end, 2:end)))

%Check if and the inverse of L are also pretty close.
L1_F_Linv_distance = max(max(F- inv(L)))

%f.
A = zeros(N-1,M+1);
A(:,1) = 1;
for n = 1:N-1   %Don't include the last timestep bc there is nothing to predict then. 
    A(n,2:2+(n-max(1,n-M+1))) = flip(r(max(1,n-M+1):n));
end

pA = pinv(A);
w_0 = pA*r(2:end);
Fm_LS_coeff_difference = F(:,1)+w_0

%g. 
abs_k = abs(k)
end
