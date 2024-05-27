function [iter,X,errList] = Complete(Tsr,w,rho1,rho2,rho3,rho4,rho5,p,lambda,beta,gamma,eta,epsilon,maxIter)
X = Tsr;
Omega = ~isnan(Tsr);
X(logical(1-Omega)) = mean(Tsr(Omega));
errList = zeros(maxIter,1);
dim = size(Tsr);
n1 = dim(1);
n2 = dim(2);
n3 = dim(3);
Tdim = ndims(Tsr);
M = cell(Tdim, 1);
Y1 = M;
for i = 1:Tdim
    M{i} = X;
    Y1{i} = zeros(dim);
end
Msum = zeros(dim);
Y1sum = zeros(dim);
K1 = 0.5*sparse(cat(1,eye(n2-2),zeros(2,n2-2)));
K2 = sparse(cat(1,zeros(1,n2-2),eye(n2-2),zeros(1,n2-2)));
K3 = 0.5*sparse(cat(1,zeros(2,n2-2),eye(n2-2)));
I_Q = eye(n1*n3);
I_D = eye(n2);
I_P = eye(n2);
I_U = eye(n1*n3);
Y4 = zeros(dim);
Y5 = zeros(dim);
Y6 = zeros(n1*n3,n2-2);
Y7 = zeros(n1*n3,n1*n3);
Z = zeros(n1*n3,n1*n3);
U = zeros(n1*n3,n1*n3);
X_tmp1 = Unfold(X,dim,2)';
P = X;
Q = X_tmp1*K2;
for iter = 1: maxIter
    rho1 =rho1 * 1.05;
    rho2 =rho2 * 1.05;
    rho3 =rho3 * 1.05;
    rho4 =rho4 * 1.05;
    rho5 =rho5 * 1.05;
    % update Mn
    Msum = 0*Msum;
    Y1sum = 0*Y1sum;
    for i = 1:Tdim
        Temp_M1 = Unfold(X+Y1{i}/rho1, dim, i);
        Temp_M2 = Unfold(M{i}, dim, i);
        [M_U,M_S,M_V] = svd(Temp_M1,'econ');
        d_m = svd(Temp_M2);
        s_m = diag(M_S);
        f = lambda*p*d_m.^(p-1)./(1+d_m.^p);
        temp = s_m-w(i)/rho1*f;
        temp(temp<0) = 0;
        Temp_M3 = M_U*diag(temp)*M_V';
        M{i} = Fold(Temp_M3,dim,i);
        Y1sum = Y1sum + Y1{i};
        Msum = Msum + M{i};
    end
    % prepare
    X_tmp = Unfold(X,dim,2)';
    % update D
    mat_d1 = Unfold(X+Y4/rho2, dim, 2);
    mat_p = Unfold(P, dim, 2);
    tmp_d = beta*K2*(eta*(K1+K3)'*mat_p+(1-eta)*Q'*U')+rho2*mat_d1;
    mat_d2 = (beta*K2*(K2)'+rho2*I_D)\tmp_d;
    D = Fold(mat_d2,dim,2);
    % update P
    mat_p1 = Unfold(X+Y5/rho3, dim, 2);
    mat_d = Unfold(D, dim, 2);
    tmp_p = beta*eta*(K1+K3)*((K2)'*mat_d-(1-eta)*Q'*U')+rho3*mat_p1;
    mat_p2 = (beta*eta*eta*(K1+K3)*(K1+K3)'+rho3*I_P)\tmp_p;
    P = Fold(mat_p2,dim,2);
    % update Q
    mat_p = Unfold(P, dim, 2);
    tmp_q1 = beta*(1-eta)^2*(U)'*U+rho4*I_Q;
    tmp_q2 = beta*(1-eta)*U'*(mat_d'*K2-eta*mat_p'*(K1+K3));
    tmp_q3 = rho4*X_tmp*K2+Y6;
    Q = tmp_q1\(tmp_q2+tmp_q3);
    % update U
    tmp_U1 = beta*(1-eta)*(mat_d'*K2-eta*mat_p'*(K1+K3))*Q'+rho5*(Z-diag(diag(Z)))-Y7;
    tmp_U2 = beta*(1-eta)^2*Q*(Q)'+rho5*I_U;
    U = tmp_U1/tmp_U2;
    % update Z
    tmp_Z1 = U+Y7/rho5;
    Z = findrootp(tmp_Z1,gamma/rho5,1);
    Z = Z-diag(diag(Z));
    % update X
    oldX = X(logical(1-Omega));
    x_t = (rho1*Msum +rho2*D+rho3*P-Y1sum-Y4-Y5)/(Tdim*rho1+rho2+rho3);
    tmp1 = Unfold(x_t,dim,2);
    tt = rho1*M{1}-Y1{1}+rho1*M{2}-Y1{2}+rho1*M{3}-Y1{3}+rho2*D-Y4+rho3*P-Y5;
    m1 = Unfold(tt, dim, 2);
    tmp2 = (K2'*m1+rho4*Q'-Y6')/(Tdim*rho1+rho2+rho3+rho4);
    [mm,~] = size(tmp1);
    tmp1(2:mm-1,:) = tmp2;
    X = Fold(tmp1,dim,2);
    X(Omega) = Tsr(Omega);
    newX = X(logical(1-Omega));
    % update Y
    for i = 1:Tdim
        Y1{i} = Y1{i} + rho1*(X-M{i});
    end
    Y4 = Y4 + rho2*(X-D);
    Y5 = Y5 + rho3*(X-P);
    X_tmp = Unfold(X,dim,2)';
    Y6 = Y6 + rho4*(X_tmp*K2-Q);
    Y7 = Y7 + rho5*(U-Z+diag(diag(Z)));
    errList(iter) = norm(newX-oldX) / norm(oldX);
    if errList(iter) < epsilon
        break;
    end
end
end  