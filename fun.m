function f = fun(x)
delT = 0.02;
n = size(x,1);
f = 0;
tau = x(:,5).^2 + x(:,6).^2;
for i=1:(n-1)
    f = f + 0.5*(tau(i)+tau(i+1))*delT;
end