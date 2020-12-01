clear; clc;
aone=[1 1];
atwo=[1 1; 1 1];
k=2; T=10;
%% Problem Data
a=4; % alpha
d=0.5; %delta
c0 = [10 1 2; 2 1 10]; %[c0(18,-2) c0(18,0) c0(18,2);c0(20,-2) c0(20,0) c0(20,2)]
c1 = [5 -1 2; 5 2 -1; -1 2 5; 2 -1 5]; %[c1(C,18,-2) c1(C,18,0) c1(C,18,2);c1(H,18,-2) c1(H,18,0) c1(H,18,2); c1(C,20,-2) c1(C,20,0) c1(C,20,2); c1(H,20,-2) c1(H,20,0) c1(H,20,2)]
p11 = [0.6 0.4; 0.2 0.8]; %[P(C|C,18) P(H|C,18); P(C|H,18) P(H|H,18)]
p12 = [0.8 0.2; 0.4 0.6]; %[P(C|C,20) P(H|C,20); P(C|H,20) P(H|H,20)]
p01 = [1 0.8 0.2; 0 0.2 0.8]; %[P(18|18,-2) P(18|18,0) P(18|18,2); P(20|18,-2) P(20|18,0) P(20|18,2)]
p02 = [0.8 0.2 0; 0.2 0.8 1]; %[P(18|20,-2) P(18|20,0) P(18|20,2); P(20|20,-2) P(20|20,0) P(20|20,2)]

%% Policy Iteration Algorithm
P_agent = [p11 p11; p12 p12];
s = zeros(4,1); % s represents value function
u0 = [3; 2; 1; 3];
u_prev = u0;
u_new = u0;
u_temp = zeros(3,1);
u_store = zeros(4,T);
u_store(:,1) = u0; 
next_iter = true;
q = false;
while next_iter
%u_prev
% Step 1
P_state = [p01(1,u_prev(1))*aone p01(2,u_prev(1))*aone; p01(1,u_prev(2))*aone p01(2,u_prev(2))*aone; ... 
    p02(1,u_prev(3))*aone p02(2,u_prev(3))*aone; p02(1,u_prev(4))*aone p02(2,u_prev(4))*aone];
A = eye(4) - d*P_agent.*P_state;
b = a*[c0(1,u_prev(1)); c0(1,u_prev(2)); c0(2,u_prev(3)); c0(2,u_prev(4))] ... 
    + [c1(1,u_prev(1)); c1(2,u_prev(2)); c1(3,u_prev(3)); c1(4,u_prev(4))];
s = A\b;
% Step 2
for j=1:4
    c0_temp = fix(j/3)+1;
    for u=1:3
        P_new = [p01(1,u)*atwo p01(2,u)*atwo; p02(1,u)*atwo p02(2,u)*atwo];
        u_temp(u) = a*c0(c0_temp,u)+c1(j,u) + d*P_agent(j,:).*P_new(j,:)*s;
    end
    [v,u_new(j)] = min(u_temp);
end
r = (u_prev == u_new);
q = true;
for i=1:4
   q = q & r(i); 
end
next_iter = ~q;
u_store(:,k)=u_new;
%u_new
u_prev=u_new;
k=k+1;
end