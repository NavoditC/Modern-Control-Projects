close all;clear;clc;

% Arm parameters
L = 1;
m = 1;
I = 1/12;

% Traj Opt parameters
dt = 0.02;
tspan = 0:dt:1.5;
N = length(tspan);
options = optimoptions('fmincon','Display', 'iter', 'MaxFunctionEvaluations', 1e5);
x0 = zeros(N,6); % % Initial condition [q1,q2,dq1,dq2,tau1,tau2]

%[M,C,N,Y] = computeDynamicsMatrices(q,dq,tau);

%%%%%%% SETUP AND SOLVE TRAJECTORY OPTIMIZATION HERE %%%%%%%%%
A = [];
b = [];
Aeq = [];
beq = [];
lb= []; 
ub= [];
%lb = [-3*pi/4*ones(N,2),-Inf*ones(N,4)];
%ub = [3*pi/4*ones(N,2),Inf*ones(N,4)];

%x = x0(:,1:2); % Replace this with the output of fmincon
[x1,fval] = fmincon(@fun,x0,A,b,Aeq,beq,lb,ub,@nonlcon,options);
x = x1(:,1:2);
disp(fval)
% X should be of size Nx2 where each column is q1,q2 at that time index
animateHW12(x,dt);