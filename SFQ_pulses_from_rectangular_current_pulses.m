%% This script generates SFQ voltage pulse train from rectangular input current pulse trains.
%% Here we use event detection in the ODE solver to detect 2 pi phase jumps.
clear all; close all; 
q = 1.6e-19; % charge of an electron 
h_bar = 1.05e-34;
factor = h_bar/(2*q); % = Phi0/(2*pi)

Ic = 1e-03; % critical current of the SIS junction (in Amps)
C = 5.7e-14; % capacitance 
Rn = 4; % Normal resistance of the junction (in Ohms) 
Rs = 2; % Shunt resistance added in parallels.
res = (1/Rn + 1/Rs); % Total effective resistance 

Ibias = 0.85*Ic; % This is the base or the starting current value that is less that Ic
Ip    = 0.2*Ic; % This is the current boots given i.e. current is increased from 0.9Ic 
% to (0.9 + 0.2)Ic = 1.1Ic. 


% Train parameters
Npulses = 3; % Number of pulses in the input current and consequently the output sfq pulses 
t_rec   = 30e-12;    % recovery time at Ibias between pulses
%lead    = 5*sig;     % time from pulse-start to gaussian center
mwin    = 6;         % safety window length in sigmas for max pulse time (Not sure how big this needs to be)

% lead is basically used to determine the peak of the next pulse. sig is
% the width of each gaussian current pulse so by the time me get to
% 1st_peak_time + 5_sig the initial pulse is almost 0 and it is here that
% we want the center of the 2nd pulse to be so that there is no overplap
% between two current pulses. 

% mwin is used to determine how long is the code supposed to wait (i.e.
% keep running at I > Ic) before it can throw an error saying no phase jump
% detected. 
% Sometimes if I > Ic is too small or of the junction is accidentally
% underdamped we would not see a full 2 pi phase shift and the phase keeps
% going back and forth in a valley and to detect such cases w use mwin *
% sig. So the code waits for mwin * sig for a phase jump to happen if not
% it throws an error. since lead is 5* sig mwin is choosen to be 6* sig.


phi_initial = asin(Ibias/Ic);
dphi_initial = 0;
y0 = [phi_initial; dphi_initial]; %initial Y_vector 

t_full = []; % This array stores all the times for the complete pulse train
I_full = []; % This array stores all the current values 
y_full = []; % this array stores the solutions of the ODE (phi and dphi) at all times 
t_events = zeros(Npulses,1); % This array is used to store the time instants when a 2pi shift happens.


t_pointer = 0; % this variable stores the time instance of we're at. 
% We start at 0 and then update the value at each step in the loop. 
rect_width = 5e-12;


%% IMPORTANT: Here we're estimating the MaxStep for the ODE options. This is not the same as the width of the rectangle because 
%% we cut the pulse off when the phase evolves exactly by 2 pi. So as an approximate estimate we use an analytical equation for
%% C = 0 case because we just need the order of magnitude and not the exact value. 
%% C = 0 analytical equation estimation. 
Phi0 = 2*pi*factor;
Ihigh = Ibias + Ip;
tau_slip_est = Phi0 / (res * sqrt(Ihigh^2 - Ic^2));
opts_stand = odeset('RelTol',1e-10,'AbsTol',1e-13,'MaxStep',tau_slip_est/50);
%opts_stand is the standard options structure which is used for ODE solving
%in both cases with and without event detection. 

for k = 1: Npulses
    t_start = t_pointer; 
    t0 = t_start; % starting point of each current pulse
    %% Rectangular pulse 
    
    I_pulse = @(t) rectangular_pulse(Ibias, Ip, t, t0, rect_width);
    phi_ref = y0(1);
    opts_events = odeset(opts_stand, 'Events', @(t,y) phase_event_2pi(t,y,phi_ref)); 
    % opts_events is the options for the ODE solver for event detection.
    % This means it has the standard options info + the info needed for
    % event detection.
    
    
    % Cap the pulse segment so it can error out if no slip happens. stop at
    % tMaxPulse.

    %% This segment solves the ODE during a current pulse with event detection
     
    %tMaxPulse = t_start + lead + mwin*sig; % This is for the gaussian
    tMaxPulse = t0 + rect_width + 10e-12; % This is for the rectangular pulse
    [t1, y1, te, ye, ie] = ode45(@(t,y) rcsj_diff_eq(t,y,I_pulse,Ic,C,factor,res), ...
                              [t_start tMaxPulse], y0, opts_events);
    if isempty(te)
            error('Pulse %d: No 2π slip detected. Increase Ip, widen sig, or increase tMaxPulse.', k);
    end
    te = te(1); % te is all the times when the 2pi jump happens.
    ye = ye(1,:).'; % storing all the y values at the 2pi phase jumps as a column vector 

    t_events(k) = te; % stores all the times when the 2pi phase jump happens
    
    % Here we make sure that we do not add the same time twice in the full
    % array. So if the full array is not empty we get rid of the redundant
    % entry.
    if ~isempty(t_full)
            t1 = t1(2:end); y1 = y1(2:end,:);
    end
        t_full = [t_full; t1];
        y_full = [y_full; y1];
        I_full = [I_full; arrayfun(I_pulse, t1)]; % computes the current value 
        % from the pulse equation for the times in t1. 

     %% Here we solve the ODE when the current is at Ibias < Ic
     % ----- Segment B: immediately drop to Ibias for recovery -----
        I_hold = @(t) Ibias;
        t2span = [te, te + t_rec]; % we start the pulse immediatly after the 2pi jump hence te
        % we use standard options because there is no need for event
        % location here. 
        % The y vector we now use is the one that corresponds to the
        % previous 2pi jump time instant. 
        [t2,y2] = ode45(@(t,y) rcsj_diff_eq(t,y,I_hold,Ic,C,factor,res), t2span, ye, opts_stand);

        t2 = t2(2:end); y2 = y2(2:end,:);% getting rid of duplicate entries

        t_full = [t_full; t2];
        y_full = [y_full; y2];
        I_full = [I_full; Ibias*ones(size(t2))];

        % Prepare for next pulse (update the pointer value to the last time
        % entry)
        t_pointer = t_full(end);
        y0 = y_full(end,:).';
end
phi  = y_full(:,1);
dphi = y_full(:,2);
V    = factor*dphi;



area = trapz(t_full, V);   % units: V*s = Weber
fprintf("Area under V(t): %.4e V-s,  Phi0: %.4e V-s,  ratio: %.6f\n", ...
        area, (factor*2*pi), area/(factor*2*pi));


subplot(3,1,1)
plot(t_full, I_full, LineWidth=2, Color='#556b2f');
xline(t_events, '--b', LineWidth=2);
xlabel('Time (s)', 'FontSize',15, Interpreter='latex');
ylabel('Current (A)', FontSize=15, Interpreter='latex');
title('Input Current Pulse', FontSize=18, Interpreter='latex');
grid on;

subplot(3,1,2)
plot(t_full, phi, LineWidth=2, Color='#483d8b');
xline(t_events, '--b', LineWidth=2);
xlabel('Time (s)', FontSize=15, Interpreter='latex');
ylabel('Junction Phase', FontSize=15, Interpreter='latex');
title('Phase Evolution', FontSize=18, Interpreter='latex');
grid on;

subplot(3,1,3)
plot(t_full, V, LineWidth=2, Color='#ff69b4');
xline(t_events, '--b', LineWidth=2);
xlabel('Time (s)', FontSize=15, Interpreter='latex');
ylabel('Voltage (v)', FontSize=15, Interpreter='latex');
title('Output SFQ Pulses', FontSize=18, Interpreter='latex');
grid on;

area = trapz(t_full, V);   % units: V*s = Weber
fprintf("Area under V(t): %.4e Wb,  Phi0: %.4e Wb,  ratio: %.6f\n", ...
        area, (factor*2*pi), area/(factor*2*pi));

%% Zooming in on a single SFQ spike 
figure;
t_event = t_events(2);
window_left  = 5e-12;
window_right = 2e-12;

idx2 = t_full >= (t_event - window_left) & t_full <= (t_event + window_right);

plot(t_full(idx2) - t_event, V(idx2), 'LineWidth', 2);
xlabel('Time relative to event (s)');
ylabel('Voltage (V)');
title('Zoomed Single SFQ Pulse');
grid on;
function I_rect = rectangular_pulse(I_bias, I_p, t, t_start, width)
    I_rect = I_bias + I_p * double(t >= t_start & t <= t_start + width);
end


%% RCSJ differential equation function 
function dy = rcsj_diff_eq(t,y,I_of_t,Ic,C,factor,res)
    I = I_of_t(t);
    dy = [ ...
        y(2); ...
        I/(C*factor) - (Ic/(C*factor))*sin(y(1)) - (1/C)*res*y(2) ...
    ];
end



%% Event Detection Function 
% This function detects when the phase starting at phi_ref (reference
% phase) becomes phi_ref + 2*pi i.e. a 2pi phase shift. 
% LOOK AT EVENT LOCATION FOR ODE DOCUMENTATION
function [value, isterminal, direction] = phase_event_2pi(~, y, phi_ref)
    value = (y(1) - phi_ref) - 2*pi;
    isterminal = 1;
    direction  = +1;
end