clear; clc;
aone = ones(2,2);
atwo = ones(2,1);
u1 = [1; 1; 0; 0];
d1 = [0; 0; 1; 1];
%% Problem Data
a=2;
d=0.5;
beta = [1; 1; 1; 1];
c0 = [10 1 2; 2 1 10]; %[c0(18,-2) c0(18,0) c0(18,2);c0(20,-2) c0(20,0) c0(20,2)]
c1 = [5 -1 2; 5 2 -1; -1 2 5; 2 -1 5]; %[c1(C,18,-2) c1(C,18,0) c1(C,18,2);c1(H,18,-2) c1(H,18,0) c1(H,18,2); c1(C,20,-2) c1(C,20,0) c1(C,20,2); c1(H,20,-2) c1(H,20,0) c1(H,20,2)]
p11 = [0.6 0.4; 0.2 0.8]; %[P(C|C,18) P(H|C,18); P(C|H,18) P(H|H,18)]
p12 = [0.8 0.2; 0.4 0.6]; %[P(C|C,20) P(H|C,20); P(C|H,20) P(H|H,20)]
p01 = [1 0.8 0.2; 0 0.2 0.8]; %[P(18|18,-2) P(18|18,0) P(18|18,2); P(20|18,-2) P(20|18,0) P(20|18,2)]
p02 = [0.8 0.2 0; 0.2 0.8 1]; %[P(18|20,-2) P(18|20,0) P(18|20,2); P(20|20,-2) P(20|20,0) P(20|20,2)]

%% Primal Program
P_agent = [p11 p11; p12 p12];
P_state = zeros(4,4,3);
A0 = zeros(4,4,3);
A = zeros(12,4);
c = zeros(4,3);
b = zeros(12,1);
I2 = [eye(4) eye(4) eye(4)];
for i=1:3
    P_state(:,:,i) = [p01(1,i)*aone p01(2,i)*aone; p02(1,i)*aone p02(2,i)*aone];
    A0(:,:,i) = P_agent.*P_state(:,:,i);
    A((4*(i-1)+1):4*(i),:) = eye(4)-d*A0(:,:,i);
    c(:,i) = a*[c0(1,i)*atwo; c0(2,i)*atwo] + c1(:,i);
    b((4*(i-1)+1):4*i) = -c(:,i);
end
s = -linprog(beta,-A,-b); % Justify -ve sign

%% Dual Program
r = zeros(4,3);
u_star = zeros(4,1);
x_star = zeros(4,1);
infeas = 0; % If 0 feasible if 1 infeasible
% for j=1:3
%    r(:,j)= -(a*c0(1,j)*u1+a*c0(2,j)*d1 +c1(:,j)); 
% end
% f2 = [r(:,1); r(:,2); r(:,3)];
A2 = -eye(12); Z = zeros(12,1);

% P_agent2 = [P_agent P_agent P_agent];
% p01_2 = [p01(1,1) p01(1,1) p01(2,1) p01(2,1) ...
%     p01(1,2) p01(1,2) p01(2,2) p01(2,2) ...
%     p01(1,3) p01(1,3) p01(2,3) p01(2,3)];
% p02_2 = [p02(1,1) p02(1,1) p02(2,1) p02(2,1) ...
%     p02(1,2) p02(1,2) p02(2,2) p02(2,2) ...
%     p02(1,3) p02(1,3) p02(2,3) p02(2,3)];
% P_state2 = [p01_2; p01_2; p02_2; p02_2];
% P_final2 = P_agent2.*P_state2;

Aeq2 = I2 - d*A';
x = linprog(-b,A2,Z,Aeq2,beta);
% Finding u_star
matx = zeros(4,3);
for q=1:3
    matx(:,q)=x(4*(q-1)+1:4*q,1);
end
for k=1:4
    [x_star(k),u_star(k)] = max(matx(k,:));
    if(x_star(k)<=0)
        infeas = 1;
    end
end
