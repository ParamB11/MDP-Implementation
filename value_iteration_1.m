clear
clc
aone=[1 1; 1 1];
%% Problem Data
a=4; % alpha
d=0.5; %delta
tol = 0.01; %tolerance for value iteration
c0 = [10 1 2; 2 1 10]; %[c0(18,-2) c0(18,0) c0(18,2);c0(20,-2) c0(20,0) c0(20,2)]
c1 = [5 -1 2; 5 2 -1; -1 2 5; 2 -1 5]; %[c1(C,18,-2) c1(C,18,0) c1(C,18,2);c1(H,18,-2) c1(H,18,0) c1(H,18,2); c1(C,20,-2) c1(C,20,0) c1(C,20,2); c1(H,20,-2) c1(H,20,0) c1(H,20,2)]
p11 = [0.6 0.4; 0.2 0.8]; %[P(C|C,18) P(H|C,18); P(C|H,18) P(H|H,18)]
p12 = [0.8 0.2; 0.4 0.6]; %[P(C|C,20) P(H|C,20); P(C|H,20) P(H|H,20)]
p01 = [1 0.8 0.2; 0 0.2 0.8]; %[P(18|18,-2) P(18|18,0) P(18|18,2); P(20|18,-2) P(20|18,0) P(20|18,2)]
p02 = [0.8 0.2 0; 0.2 0.8 1]; %[P(18|20,-2) P(18|20,0) P(18|20,2); P(20|20,-2) P(20|20,0) P(20|20,2)]

%% Value Iteration Algorithm 
P_agent = [p11 p11; p12 p12];
s_prev = zeros(4,1); % s represents value function
s_new = zeros(4,1);
s_temp = zeros(3,1);
u_star = zeros(4,1);
next_iter = true;
temp = true;
while next_iter %for t=1:N
    for j=1:4
    c0_temp = fix(j/3)+1;
        for u=1:3
            P_state = [p01(1,u)*aone p01(2,u)*aone; p02(1,u)*aone p02(2,u)*aone];
            s_temp(u) = a*c0(c0_temp,u)+c1(j,u)+d*P_agent(j,:).*P_state(j,:)*s_prev;
        end
    s_new(j) = min(s_temp);
    end
    next_iter=false;
    for i=1:4
        temp=abs(s_prev(i)-s_new(i))>=tol;
        next_iter=next_iter|temp;
    end
    s_prev=s_new;
    %s(:,t)=s_prev;
end
s_star=s_new;
for j=1:4
c0_temp = fix(j/3)+1;
for u=1:3
    P_state = [p01(1,u)*aone p01(2,u)*aone; p02(1,u)*aone p02(2,u)*aone];
    s_temp(u) = a*c0(c0_temp,u)+c1(j,u)+d*P_agent(j,:).*P_state(j,:)*s_star;
end
[v,u_star(j)] = min(s_temp);
end

% time = linspace(1,N,N);
% figure('Name','Value Function Components');
% plot(time,s(1,(1:N)),time,s(2,(1:N)),time,s(3,(1:N)),'--',time,s(4,(1:N)),'--','LineWidth',2);
% legend({'s1','s2','s3','s4'},'FontSize',15);