function [c,ceq] = nonlcon(x)
delT = 0.02;
n = size(x,1);
q = x(:,1:2);
dq = x(:,3:4);
tau = x(:,5:6);
q_diff = [];
dq_diff = [];
for i=1:(n-1)
    q_diff = [q_diff,q(i+1,:)-(q(i,:)+delT*0.5*(dq(i,:)+dq(i+1,:)))];
    [M,C,N,Y] = computeDynamicMatrices(q(i,:)',dq(i,:)',tau(i,:)');
    ddq_curr = inv(M)*(Y-C*dq(i,:)'-N);
    [M,C,N,Y] = computeDynamicMatrices(q(i+1,:)',dq(i+1,:)',tau(i+1,:)');
    ddq_next = inv(M)*(Y-C*dq(i+1,:)'-N);
    dq_diff = [dq_diff,dq(i+1,:)-(dq(i,:)+delT*0.5*(ddq_curr'+ddq_next'))];
end
% Nonlinear equality constraints
ceq = [q_diff,dq_diff,q(1,:)-[-pi/2 0],dq(1,:),q(end,:)-[pi/2 0],dq(end,:)];
    
% Nonlinear inequality constraints
c = [];

end