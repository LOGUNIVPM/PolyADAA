% PolyADAA - a novel method for aliasing reduction in memoryless nonlinearities
% Author: Leonardo Gabrielli <l.gabrielli@staff.univpm.it>
%
% This code accompanies the submission of the paper 
% "PolyADAA: Improving Aliasing Reduction in Memoryless Nonlinearities
% Using Lagrange Interpolation and Polynomial Approximation"
% to appear at the 29th Int. Conference on Digital Audio Effects (DAFx26),
% Cambridge MA, USA

% The code shows how to perform aliasing reduction using triangular or
% rectangular filter kernels with a tanh or hardclip nonlinearity.

% The code is free to use but please mention its author. To cite this
% research in your paper see the bibtex entry in the README.md file.

Fs = 48000;
N  = Fs;            % duration
f0 = 3192;          % sine tone pitch
A  = 9;             % sine tone peak
K  = 32;            % Chebyshev order
L_ord = 3;          % Lagrange order
kernel = 'tri';     % 'tri' or 'rect'
nonlintype = 'tanh'; % you can also try 'hardclip'

if strcmp(nonlintype, 'tanh')
    f = @(x) tanh(x);                   % nonlinearity
    F0 = @(x) log(cosh(x));             % antiderivative for linear-ADAA
    F1 = @(x) arrayfun(@(xx) ...        % antiderivative of x*tanh(x)
        (integral(@(t) t.*tanh(t), 0, xx)), x);
end
if strcmp(nonlintype, 'hardclip')
    f = @(x) 0.5 * (abs(x+1)-abs(x-1));     % hard-clip nonlinearity as defined by Bilbao
    F0 = @(x) 1/4 * ((x+1).^2*sign(x+1)-(x-1).^2*sign(x-1)-2); % antiderivative for linear-ADAA
    F1 = @(x) arrayfun(@(xx) ...    % antiderivative of x*f(x)
        (integral(@(t) t.*( 0.5 * (abs(x+1)-abs(x-1)) ), 0, xx)), x);
end

% sine input
n = (0:N-1).';
x = A * sin(2*pi*f0*n/Fs);

% Outputs
y_triv   = zeros(N,1);
y_polyADAA = zeros(N,1);

%% Chebyshev nodes and mapping
tau_nodes = cos( (0:K)' * pi/K );     % nodes in [-1,1]
map_tau = @(t) 0.5*(t+1);             % map [-1,1] -> [0,1]
tau_phys = map_tau(tau_nodes);        % nodes in [0,1]

%% Precompute Chebyshev integrals M_k

% Chebyshev integration grid (dense for accuracy)
Nt = 2000;
t  = linspace(-1,1,Nt);
dt = t(2) - t(1);

% Select kernel in Chebyshev variable t
switch kernel
    case 'rect'
        W = 0.5 * ones(size(t));            % rectangular kernel
    case 'tri'
        W = 2 * (1 - abs(t));               % triangular kernel
    otherwise
        error('Unknown kernel type');
end

% Precompute Chebyshev moments M_k
M = zeros(K+1,1);
for k = 0:K
    Tk = cos(k * acos(t));                  % Chebyshev polynomial T_k(t)
    M(k+1) = sum(Tk .* W) * dt;             % numerical integration
end


%% MAIN LOOP: sample by sample processing

for n0 = 3:N-2
    % trivial method
    y_triv(n0) = f( x(n0) );
    
    % propose 
    if L_ord == 2
        % Build quadratic interpolant across tau in [0,1] using x[n0-1], x[n0], x[n0+1]
        xv = [ x(n0-1); x(n0); x(n0+1) ];
        Tpos = [0; 0.5; 1];                % sample positions in tau ∈ [0,1]
        p = polyfit(Tpos, xv, 2);         % p(1)*tau^2 + p(2)*tau + p(3)
        xtau = @(tau) polyval(p, tau);
    elseif L_ord == 3
        % cubic (order 3) Lagrange interpolation
        xv = [ x(n0-1); x(n0); x(n0+1); x(n0+2) ];
        Tpos = [0; 1/3; 2/3; 1];
        p = polyfit(Tpos, xv, 3);
        xtau = @(tau) polyval(p, tau);
    else
        error('Wrong Lagrange interpolation order');
    end

    % sample g at Chebyshev nodes in [0,1]
    gvals = f( xtau(tau_phys) );

    % compute Chebyshev coefficients c_k via DCT-I trick
    bigvec = [gvals; gvals(end-1:-1:2)];
    Cfft = real( fft(bigvec) );
    c = Cfft(1:K+1) / K;
    c(1) = c(1) / 2;
    c(end) = c(end) / 2;

    % weighted sum using M
    y_polyADAA(n0) = sum( c .* M );
end

y_polyADAA = y_polyADAA / 2; % fix amp

%% PLOT OUTPUTS

figure; 
plot(1:100, y_triv(1:100)); hold on;
plot(1:100, y_polyADAA(1:100),'r');
legend('trivial','PolyADAA');
title('trivial vs PolyADAA')

lastHarmBeforeNyquist = floor(floor((Fs/2) / f0) ) + 1; 

figure,
snr(y_triv, Fs, lastHarmBeforeNyquist, 'omitaliases');  
triv_snr = snr(y_triv, Fs, lastHarmBeforeNyquist, 'omitaliases');  
ylim([-100,0]);
tit = sprintf('Trivial SNR: %0.2f', triv_snr);
title(tit);

figure,
snr(y_polyADAA, Fs, lastHarmBeforeNyquist, 'omitaliases');  
poly_snr = snr(y_polyADAA, Fs, lastHarmBeforeNyquist, 'omitaliases');  
ylim([-100,0]);
tit = sprintf('PolyADAA SNR: %0.2f', poly_snr);
title(tit);