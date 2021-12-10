function [M,C,N,Y] = computeDynamicMatrices(in1,in2,in3)
%COMPUTEDYNAMICMATRICES
%    [M,C,N,Y] = COMPUTEDYNAMICMATRICES(IN1,IN2,IN3)

%    This function was generated by the Symbolic Math Toolbox version 8.1.
%    06-Nov-2018 14:37:13

dq1 = in2(1,:);
dq2 = in2(2,:);
q1 = in1(1,:);
q2 = in1(2,:);
tau_1 = in3(1,:);
tau_2 = in3(2,:);
t2 = cos(q2);
t3 = t2.*(1.0./2.0);
t4 = t3+1.0./3.0;
M = reshape([t2+5.0./3.0,t4,t4,1.0./3.0],[2,2]);
if nargout > 1
    t5 = sin(q2);
    C = reshape([dq2.*t5.*(-1.0./2.0),dq1.*t5.*(1.0./2.0),t5.*(dq1+dq2).*(-1.0./2.0),0.0],[2,2]);
end
if nargout > 2
    t6 = q1+q2;
    t7 = cos(t6);
    t8 = t7.*(9.81e2./2.0e2);
    N = [t8+cos(q1).*1.4715e1;t8];
end
if nargout > 3
    Y = [tau_1;tau_2];
end
